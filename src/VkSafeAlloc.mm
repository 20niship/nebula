// ObjC++ wrapper for VMA calls that may throw NSException on AppleParavirtDevice.
// Provides tryVmaCreateBuffer — declared in AttributeBuffer.cpp on Apple builds.
#import <Foundation/Foundation.h>
#include <vk_mem_alloc.h>
#include <cstdio>

extern "C" VkResult
tryVmaCreateBuffer(VmaAllocator                   allocator,
                   const VkBufferCreateInfo*       pBufferCreateInfo,
                   const VmaAllocationCreateInfo*  pAllocationCreateInfo,
                   VkBuffer*                       pBuffer,
                   VmaAllocation*                  pAllocation,
                   VmaAllocationInfo*              pAllocationInfo)
{
    __block VkResult result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    @try {
        result = vmaCreateBuffer(allocator, pBufferCreateInfo, pAllocationCreateInfo,
                                 pBuffer, pAllocation, pAllocationInfo);
    } @catch (NSException* ex) {
        fprintf(stderr, "[VkWrapper] NSException in vmaCreateBuffer: %s\n",
                [[ex reason] UTF8String] ?: "(null)"); fflush(stderr);
        if (pBuffer)     *pBuffer     = VK_NULL_HANDLE;
        if (pAllocation) *pAllocation = VK_NULL_HANDLE;
    } @catch (...) {
        fprintf(stderr, "[VkWrapper] unknown exception in vmaCreateBuffer\n"); fflush(stderr);
        if (pBuffer)     *pBuffer     = VK_NULL_HANDLE;
        if (pAllocation) *pAllocation = VK_NULL_HANDLE;
    }
    return result;
}
