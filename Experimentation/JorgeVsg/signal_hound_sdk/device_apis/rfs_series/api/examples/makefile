C = gcc
CFLAGS = -Wall -g

OBJECTS = main.o rfs_api.o

main: $(OBJECTS)
		@$(CC) $(CFLAGS) -o $@ $^

$OBJECTS: rfs_api.h
