CC = gcc
CFLAGS = -lpthread
SOURCES = parallel_exec
TARGETS = $(SOURCES:.c=)

all: $(TARGETS)

%: %.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f $(TARGETS)
