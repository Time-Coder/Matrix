CC = D:/MinGW-W64/bin/g++
CC_FLAGS = -std=c++11
LINK_FLAGS = -LD:/MinGW-W64/lib64
TARGET = bin/main.exe

# EXTERNAL_INCLUDES = \

# EXTERNAL_STATIC_LIBS = \

# EXTERNAL_DYNAMIC_LIBS = \

MSYS_PATH = D:/Git/usr/bin
DIRS = $(patsubst ./%, %, $(shell $(MSYS_PATH)/find . -type d -not -path "*/.*"))

C_SRCS = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.c))
CPP_SRCS = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
SRCS = $(patsubst ./%, %, $(C_SRCS) $(CPP_SRCS))
INCLUDES = $(addprefix -I, $(DIRS)) $(addprefix -I, $(EXTERNAL_INCLUDES))

STATIC_LIBS = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.a)) \
              $(foreach dir, $(DIRS), $(wildcard $(dir)/*.lib))
DYNAMIC_LIBS = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.dll)) \
               $(foreach dir, $(DIRS), $(wildcard $(dir)/*.so))
LIB_FILES = $(STATIC_LIBS) $(DYNAMIC_LIBS) $(EXTERNAL_STATIC_LIBS) $(EXTERNAL_DYNAMIC_LIBS)
LIB_PATHS = $(foreach dir, $(dir $(LIB_FILES)), -L$(dir))
LIB_LINKS = $(foreach file, $(patsubst lib%,%,$(basename $(notdir $(STATIC_LIBS) $(EXTERNAL_STATIC_LIBS)))), -l$(file))
LIBS = $(LIB_PATHS) $(LIB_LINKS)

OBJS = $(foreach src, $(SRCS), bin/$(src).o)
DEPS = $(patsubst %.o, %.d, $(OBJS))

.PHONY: all
all: $(TARGET)
	
$(TARGET): $(OBJS)
	@$(MSYS_PATH)/mkdir -p $(dir $@)
	@echo Linking $(notdir $@)
	@$(CC) $(LINK_FLAGS) $^ $(LIBS) -o $@

bin/%.o: %
	@$(MSYS_PATH)/mkdir -p $(dir $@)
	@echo Compiling $(notdir $<)
	@$(CC) -c $(CC_FLAGS) $(INCLUDES) $< -o $(patsubst %.d,%.o,$@) -MMD -MP -MT $@\ $(patsubst %.o,%.d,$@)

bin/%.d: %
	@$(MSYS_PATH)/mkdir -p $(dir $@)
	@echo Compiling $(notdir $<)
	@$(CC) -c $(CC_FLAGS) $(INCLUDES) $< -o $(patsubst %.d,%.o,$@) -MMD -MP -MT $@\ $(patsubst %.d,%.o,$@)

-include $(DEPS)

.PHONY: clean
clean:
	$(MSYS_PATH)/rm -rf bin

.PHONY: run
run: $(TARGET)
	$<