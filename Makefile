objects = main.o bp.o
compiler= g++
cflags  = -c -Wall

bpnet : $(objects)
	$(compiler) -o bpnet $(objects)
main.o: main.cpp
	$(compiler) $(cflags) main.cpp
bp.o  : bp.cpp
	$(compiler) $(cflags) bp.cpp

clean :
	-rm -f bpnet
	-rm -f *.o
