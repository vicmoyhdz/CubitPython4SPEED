# CubitPython4SPEED
The meshing tool presented here is inspired from GEOCUBIT (https://github.com/casarotti/GEOCUBIT--experimental/tree/master/geocubitlib). 
The routines have been modified to be compatible with Python3 and with the latest version of CUBIT.
Furthermore, routines for the creation of SPEED (https://speed.mox.polimi.it/) input files are included.

READ: CubitPython4SPEED Manual.pdf

UPDATE 2026: Running Cubit from an external Python script as shown in page 10 of the manual can produce Python compatibility problems. A safer option is to use Cubit’s bundled Python as the interpreter. You can do this as: 
"C:\Program Files\Coreform Cubit 2025.12\bin\python3\python.exe" script.py
