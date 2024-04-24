# Makefile for IPA_projekt_2024

CXX = g++
CXXFLAGS = -g -std=c++11 -fpic `pkg-config --cflags opencv4`
LDFLAGS = -ldl
LIBS = `pkg-config --cflags --libs opencv4`
INCLUDES = -Iinclude/

SRCS = retinanetpost.cpp src/utils.cpp src/prior_boxes.cpp src/reader.cpp
ASMSRC = retinanetpost_asm.s
OBJS = $(SRCS:.cpp=.o) $(ASMSRC:.s=.o)
TARGET = retinanetpost
#DEBUG = -DDEBUG

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(DEBUG)  -o $@ $^ $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEBUG)  $(INCLUDES) -c $< -o $@

%.o: %.s
	gcc -masm=intel -g -c $< -o $@

clean:
	$(RM) $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET) input/input.jpg input/input.txt