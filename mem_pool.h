//
// Created by vikimaster2 on 9/24/24.
//

#ifndef CUDA_LSYSTEM_CPP_DEBUG_MEM_POOL_H
#define CUDA_LSYSTEM_CPP_DEBUG_MEM_POOL_H

#include <vector>
#include <cstddef>
#include <new>
#include <stdexcept>

template<typename T>
class MemoryPool {
private:
    std::mutex mutex;
    struct Block {
        Block* next;
    };

    std::vector<char*> chunks;
    Block* free_list;
    std::size_t chunk_size;
    std::size_t objects_per_chunk;

public:
    MemoryPool(std::size_t objects_per_chunk = 1024)
            : free_list(nullptr),
              chunk_size(sizeof(T) > sizeof(Block) ? sizeof(T) : sizeof(Block)),
              objects_per_chunk(objects_per_chunk)
    {
        allocate_chunk();
    }

    ~MemoryPool() {
        for (char* chunk : chunks) {
            delete[] chunk;
        }
    }

    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex);
        if (free_list == nullptr) {
            allocate_chunk();
        }

        Block* block = free_list;
        free_list = block->next;
        return reinterpret_cast<T*>(block);
    }

    void deallocate(T* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        if (ptr == nullptr) return;

        // Check if the pointer belongs to one of our chunks
        bool ptr_in_chunk = false;
        for (const auto& chunk : chunks) {
            if (ptr >= reinterpret_cast<T*>(chunk) &&
                ptr < reinterpret_cast<T*>(chunk + chunk_size * objects_per_chunk)) {
                ptr_in_chunk = true;
                break;
            }
        }

        if (!ptr_in_chunk) {
            throw std::invalid_argument("Attempt to deallocate memory not owned by this MemoryPool");
        }

        Block* block = reinterpret_cast<Block*>(ptr);
        block->next = free_list;
        free_list = block;
    }

    template<typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        new (ptr) T(std::forward<Args>(args)...);
        return ptr;
    }

    void destroy(T* ptr) {
        if (ptr == nullptr) return;
        ptr->~T();
        deallocate(ptr);
    }

    void resize(std::size_t objects_per_chunk) {
        this->objects_per_chunk = objects_per_chunk;

        for (char* chunk : chunks) {
            delete[] chunk;
        }

        chunks.clear();

        free_list = nullptr;
        allocate_chunk();
    }

private:
    void allocate_chunk() {
        std::lock_guard<std::mutex> lock(mutex);
        char* chunk = new char[chunk_size * objects_per_chunk];
        chunks.push_back(chunk);

        for (std::size_t i = 0; i < objects_per_chunk - 1; ++i) {
            Block* block = reinterpret_cast<Block*>(chunk + i * chunk_size);
            block->next = reinterpret_cast<Block*>(chunk + (i + 1) * chunk_size);
        }

        Block* last_block = reinterpret_cast<Block*>(chunk + (objects_per_chunk - 1) * chunk_size);
        last_block->next = free_list;  // Connect to the existing free list
        free_list = reinterpret_cast<Block*>(chunk);
    }


};

#endif //CUDA_LSYSTEM_CPP_DEBUG_MEM_POOL_H
