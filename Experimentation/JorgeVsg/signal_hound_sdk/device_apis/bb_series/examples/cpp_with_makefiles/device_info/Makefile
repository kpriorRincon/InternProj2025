CC=g++
COPTS=-Wall
EXE=dev_info

all:
	$(CC) $(COPTS) main.cpp -o $(EXE) -Wl,-rpath /usr/local/lib -lbb_api

clean:
	rm -f *~ $(EXE)
