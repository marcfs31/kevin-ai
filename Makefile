# small-llm-in-c. Sources are discovered by wildcard: adding a file under src/ or a
# tests/unit/test_*.c needs no Makefile edit.
#
#   make            build bin/* (one binary per src/cli/*.c)
#   make test       build and run every tests/unit/test_*.c
#   make gate       compare bin/lm-run against the Python oracle's golden files
#   make sanitize   run the unit tests under ASan + UBSan
#   make portable   x86-64 build with -mavx2 -mfma instead of -march=native

CC      ?= cc
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),x86_64)
  ARCH ?= -march=native
else
  ARCH ?= -mcpu=native
endif

# Apple Clang ships no OpenMP runtime; there the pragmas are ignored and the build is
# single-threaded but numerically identical.
ifeq ($(UNAME_S),Linux)
  OMP_CFLAGS  ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif

WARN     := -Wall -Wextra -Wno-unknown-pragmas
OPT      ?= -O2 -g
CFLAGS   ?= $(OPT) -std=gnu99 $(WARN) $(ARCH) $(OMP_CFLAGS)
CPPFLAGS += -Iinclude -Isrc -Ithird_party
LDLIBS   += -lm $(OMP_LDFLAGS)

B ?= build

LIB_SRCS  := $(wildcard src/core/*.c src/io/*.c src/cache/*.c src/model/*.c src/tokenizer/*.c src/chat/*.c)
LIB_OBJS  := $(LIB_SRCS:%.c=$(B)/%.o)
CLI_SRCS  := $(wildcard src/cli/*.c)
BINS      := $(CLI_SRCS:src/cli/%.c=bin/%)
TEST_SRCS := $(wildcard tests/unit/test_*.c)
TESTS     := $(TEST_SRCS:tests/unit/%.c=$(B)/tests/unit/%)
TEST_SH   := $(wildcard tests/scripts/test_*.sh)

.PHONY: all test gate gate-fast sanitize portable clean
.SECONDARY:

all: $(BINS)

$(B)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

bin/%: $(B)/src/cli/%.o $(LIB_OBJS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(B)/tests/unit/%: $(B)/tests/unit/%.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

# A test target with nothing to run must fail, not pass.
test: $(TESTS) bin/lm-run
	@test -n "$(TESTS)" || { echo "make test: no tests/unit/test_*.c found"; exit 1; }
	@fail=0; for t in $(TESTS); do echo "== $$t"; ./$$t || { echo "FAILED: $$t"; fail=1; }; done; \
	for s in $(TEST_SH); do echo "== $$s"; bash $$s || { echo "FAILED: $$s"; fail=1; }; done; exit $$fail

gate: bin/lm-run bin/lm-quant
	python3 tools/check_golden.py --run bin/lm-run --quant bin/lm-quant --fixtures tests/fixtures/qwen35-tiny
	python3 tools/check_golden.py --run bin/lm-run --quant bin/lm-quant --fixtures tests/fixtures/gemma4-tiny

# Not a gate: how far the opt-in int8 fast path is from the q8 oracle (see README, --fast).
gate-fast: bin/lm-run bin/lm-quant
	python3 tools/check_golden.py --fast --run bin/lm-run --quant bin/lm-quant --fixtures tests/fixtures/qwen35-tiny
	python3 tools/check_golden.py --fast --run bin/lm-run --quant bin/lm-quant --fixtures tests/fixtures/gemma4-tiny

sanitize:
	$(MAKE) B=build-san OPT="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" test

portable:
	$(MAKE) ARCH="-mavx2 -mfma" all

clean:
	find build build-san bin -delete 2>/dev/null || true

-include $(shell find $(B) -name '*.d' 2>/dev/null)
