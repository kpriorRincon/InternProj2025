CC=g++
COPTS=-Wall
EXE=peak_search

all:
	$(CC) $(COPTS) main.cpp -o $(EXE) -Wl,-rpath /usr/local/lib/ -lbb_api

clean:
	rm -f *~ $(EXE)
