all: main.exe

main.exe: main.o matrixf.o
	g++ -std=c++11 *.o -o main.exe

main.o: main.cpp matrix.h matrix.cpp
	g++ -std=c++11 -c main.cpp

matrixf.o: matrixf.cpp matrix.h
	g++ -std=c++11 -c matrixf.cpp

run: main.exe
	./main.exe

clean:
	del *.o >nul 2>&1
	del *.exe >nul 2>&1

clear:
	del *.o >nul 2>&1