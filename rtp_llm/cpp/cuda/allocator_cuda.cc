#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sys/mman.h>

namespace rtp_llm {

void* ICudaAllocator::reMalloc(void* ptr, size_t size) {
    size              = ((size + 127) / 128) * 128;  // make the buffer align with 128 bytes
    void* void_ptr    = (void*)ptr;
    void* ptr_address = void_ptr;
    if (isExist(ptr_address)) {
        ReallocType realloc_type = isReMalloc(ptr_address, size);
        if (realloc_type == ReallocType::INCREASE) {
            RTP_LLM_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size);
        } else if (realloc_type == ReallocType::DECREASE) {
            RTP_LLM_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size);
        } else {
            RTP_LLM_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", void_ptr, size);
            return void_ptr;
        }
    } else {
        RTP_LLM_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
        return malloc(size);
    }
}

PurePointerCudaAllocator::PurePointerCudaAllocator(int device_id):
    ICudaAllocator(device_id), pointer_mapping_(new std::unordered_map<void*, size_t>) {}

PurePointerCudaAllocator::~PurePointerCudaAllocator() {}

void PurePointerCudaAllocator::destroy() {
    while (!pointer_mapping_->empty()) {
        auto it  = pointer_mapping_->begin();
        auto ptr = it->first;
        free(&ptr);
    }
}

bool PurePointerCudaAllocator::isExist(void* address) const {
    return pointer_mapping_->count(address) > 0;
}

ReallocType PurePointerCudaAllocator::isReMalloc(void* address, size_t size) const {
    RTP_LLM_CHECK(isExist(address));
    if (pointer_mapping_->at(address) < size) {
        return ReallocType::INCREASE;
    } else if (pointer_mapping_->at(address) == size) {
        return ReallocType::REUSE;
    } else {
        return ReallocType::DECREASE;
    }
}

void* PurePointerCudaAllocator::malloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void*                       ptr = doMalloc(size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
    return ptr;
}

void* PurePointerCudaAllocator::mallocSync(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void*                       ptr = doMallocSync(size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
    return ptr;
}

void PurePointerCudaAllocator::free(void** ptr) {
    void* address = *ptr;
    if (address) {
        std::lock_guard<std::mutex> lock(lock_);
        RTP_LLM_CHECK_WITH_INFO(
            pointer_mapping_->count(address), "pointer_mapping_ does not have information of ptr at %p", address);
        doFree(address);
        *ptr = nullptr;
        pointer_mapping_->erase(address);
    }
    return;
}

Allocator<AllocatorType::CUDA>::Allocator(int device_id): PurePointerCudaAllocator(device_id) {}

Allocator<AllocatorType::CUDA>::~Allocator() {
    destroy();
}

void* Allocator<AllocatorType::CUDA>::doMalloc(size_t size) {
    void* ptr = nullptr;
    check_cuda_value(cudaMalloc(&ptr, (size_t)(ceil(size / 128.)) * 128));
    return ptr;
}

void* Allocator<AllocatorType::CUDA>::doMallocSync(size_t size) {
    void* ptr = nullptr;
    check_cuda_value(cudaMalloc(&ptr, (size_t)(ceil(size / 128.)) * 128));
    return ptr;
}

void Allocator<AllocatorType::CUDA>::doFree(void* address) {
    // tmp sync to avoid memory free before kernel run. cudaFree will not perform any implicit synchronization when the
    // pointer was allocated with cudaMallocAsync or cudaMallocFromPoolAsync
    check_cuda_value(cudaStreamSynchronize(stream_));
    check_cuda_value(cudaFree(address));
    return;
}

Allocator<AllocatorType::CUDA_HOST>::Allocator(int device_id): PurePointerCudaAllocator(device_id) {
    const char* env          = std::getenv("CUDA_HOST_THP_THRESHOLD_MB");
    size_t      threshold_mb = 256;  // default 256 MB
    if (env) {
        char* end = nullptr;
        long  val = std::strtol(env, &end, 10);
        if (end != env && val >= 0) {
            threshold_mb = static_cast<size_t>(val);
        }
    }
    thp_threshold_bytes_ = threshold_mb * 1024UL * 1024UL;
    RTP_LLM_LOG_INFO("CUDA_HOST allocator THP threshold: %zu MB (%zu bytes), %s",
                     threshold_mb,
                     thp_threshold_bytes_,
                     threshold_mb > 0 ? "THP optimization enabled for large allocations" : "THP optimization disabled");
}

Allocator<AllocatorType::CUDA_HOST>::~Allocator() {
    destroy();
}

void* Allocator<AllocatorType::CUDA_HOST>::doMalloc(size_t size) {
    size = (size_t)(ceil(size / 128.)) * 128;
    if (thp_threshold_bytes_ > 0 && size >= thp_threshold_bytes_) {
        void* ptr = doMallocWithTHP(size);
        if (ptr) {
            return ptr;
        }
        RTP_LLM_LOG_WARNING("THP allocation failed for size %zu, falling back to cudaMallocHost", size);
    }
    void* ptr = nullptr;
    check_cuda_value(cudaMallocHost(&ptr, size));
    return ptr;
}

void* Allocator<AllocatorType::CUDA_HOST>::doMallocWithTHP(size_t size) {
    constexpr size_t HUGE_PAGE_SIZE = 2UL * 1024 * 1024;  // 2MB
    size_t           aligned_size   = ((size + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE;

    RTP_LLM_LOG_INFO("THP allocation: requested %zu bytes, aligned to %zu bytes (%zu huge pages)",
                     size,
                     aligned_size,
                     aligned_size / HUGE_PAGE_SIZE);

    // Step 1: mmap to allocate virtual address space
    void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        RTP_LLM_LOG_WARNING("mmap failed for THP allocation, size=%zu: %s", aligned_size, strerror(errno));
        return nullptr;
    }

    // Step 2: advise kernel to use transparent huge pages
    if (madvise(ptr, aligned_size, MADV_HUGEPAGE) != 0) {
        RTP_LLM_LOG_WARNING("madvise MADV_HUGEPAGE failed (non-fatal), size=%zu: %s", aligned_size, strerror(errno));
        // Continue anyway - kernel will fall back to small pages
    }

    // Step 3: touch all pages to trigger page faults (using huge pages if available)
    memset(ptr, 0, aligned_size);

    // Step 4: register as CUDA pinned memory
    cudaError_t err = cudaHostRegister(ptr, aligned_size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING("cudaHostRegister failed for THP allocation: %s, falling back", cudaGetErrorString(err));
        munmap(ptr, aligned_size);
        return nullptr;
    }

    // Record the THP allocation for proper cleanup
    {
        std::lock_guard<std::mutex> lock(thp_mu_);
        thp_allocations_[ptr] = aligned_size;
    }

    RTP_LLM_LOG_INFO("THP allocation succeeded: ptr=%p, size=%zu bytes", ptr, aligned_size);
    return ptr;
}

void* Allocator<AllocatorType::CUDA_HOST>::doMallocSync(size_t size) {
    return doMalloc(size);
}

void Allocator<AllocatorType::CUDA_HOST>::doFree(void* address) {
    if (!address) {
        return;
    }

    size_t thp_size = 0;
    {
        std::lock_guard<std::mutex> lock(thp_mu_);
        auto                        it = thp_allocations_.find(address);
        if (it != thp_allocations_.end()) {
            thp_size = it->second;
            thp_allocations_.erase(it);
        }
    }

    if (thp_size > 0) {
        // THP path: unregister from CUDA and unmap
        cudaError_t err = cudaHostUnregister(address);
        if (err != cudaSuccess) {
            RTP_LLM_LOG_WARNING("cudaHostUnregister failed: %s", cudaGetErrorString(err));
        }
        if (munmap(address, thp_size) != 0) {
            RTP_LLM_LOG_WARNING("munmap failed: %s", strerror(errno));
        }
    } else {
        // Original path
        check_cuda_value(cudaFreeHost(address));
    }
}

}  // namespace rtp_llm
