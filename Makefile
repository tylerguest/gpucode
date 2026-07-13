NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17

BIN_DIR := bin

.PHONY: all clean vecadd saxpy reduce_sum

all: vecadd saxpy reduce_sum

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

vecadd: $(BIN_DIR)/vecadd
reduce_sum: $(BIN_DIR)/reduce_sum


$(BIN_DIR)/vecadd: kernels/vecadd.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BIN_DIR)/reduce_sum: tests/test_reduce_sum.cu kernels/reduce_sum.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

saxpy: $(BIN_DIR)/saxpy

$(BIN_DIR)/saxpy: tests/test_saxpy.cu kernels/saxpy.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -rf $(BIN_DIR)
