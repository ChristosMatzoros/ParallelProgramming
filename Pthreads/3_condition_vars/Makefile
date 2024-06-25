CC = gcc
CFLAGS = -lpthread
SOURCES = producer_consumer basic
TARGETS = $(SOURCES:.c=)

all: $(TARGETS)

%: %.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f $(TARGETS)
