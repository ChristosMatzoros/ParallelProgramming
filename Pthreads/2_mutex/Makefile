CC = gcc
CFLAGS = -lpthread
SOURCES = race_condition.c with_mutex.c
TARGETS = $(SOURCES:.c=)

all: $(TARGETS)

%: %.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f $(TARGETS)
