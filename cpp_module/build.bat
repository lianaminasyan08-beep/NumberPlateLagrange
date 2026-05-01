@echo off
echo Building C++ Lagrange DLL...

g++ -shared -o liblagrange.dll lagrange_core.cpp -O2 -std=c++11

echo Done.
pause