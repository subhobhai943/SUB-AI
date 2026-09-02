#
# Makefile — Builds the SUB-AI C inference engine.
#

CC     = gcc
CFLAGS = -O2 -Wall -std=c99
LDFLAGS = -lm
SRCS   = inference.c loader.c matmul.c kvcache.c model.c sampler.c tokenizer.c
OBJS   = $(SRCS:.c=.o)

all: inference

inference: $(OBJS)
	$(CC) $(CFLAGS) -o inference $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) inference

.PHONY: all clean
