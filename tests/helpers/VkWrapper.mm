// Objective-C++ wrapper: catches NSException thrown by MoltenVK/Metal
// and converts it to a C++ std::runtime_error so doctest can report details.
#import <Foundation/Foundation.h>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>

extern "C" VkResult
tryVkCreateInstance(const VkInstanceCreateInfo*  pCI,
                    const VkAllocationCallbacks* pAllocator,
                    VkInstance*                  pInstance)
{
    __block VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    @try {
        result = vkCreateInstance(pCI, pAllocator, pInstance);
    } @catch (NSException* ex) {
        std::string msg = "NSException in vkCreateInstance — name: ";
        msg += [[ex name]   UTF8String] ?: "(null)";
        msg += "  reason: ";
        msg += [[ex reason] UTF8String] ?: "(null)";
        fprintf(stderr, "[VkWrapper] %s\n", msg.c_str()); fflush(stderr);
        throw std::runtime_error(msg);
    } @catch (...) {
        fprintf(stderr, "[VkWrapper] unknown non-ObjC exception in vkCreateInstance\n");
        fflush(stderr);
        throw;
    }
    return result;
}
