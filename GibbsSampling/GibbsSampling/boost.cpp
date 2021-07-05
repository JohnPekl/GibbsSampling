/*
4.1   Build From the Visual Studio IDE
From Visual Studio's File menu, select New > Project��
In the left-hand pane of the resulting New Project dialog, select Visual C++ > Win32.
In the right-hand pane, select Win32 Console Application (VS8.0) or Win32 Console Project (VS7.1).
In the name field, enter ��example��
Right-click example in the Solution Explorer pane and select Properties from the resulting pop-up menu
In Configuration Properties > C/C++ > General > Additional Include Directories, enter the path to the Boost root directory, for example
C:\Program Files\boost\boost_1_55_0
In Configuration Properties > C/C++ > Precompiled Headers, change Use Precompiled Header (/Yu) to Not Using Precompiled Headers.2
Replace the contents of the example.cpp generated by the IDE with the example code above.

6.1   Link From Within the Visual Studio IDE
Starting with the header-only example project we created earlier:
Right-click example in the Solution Explorer pane and select Properties from the resulting pop-up menu
In Configuration Properties > Linker > Additional Library Directories, enter the path to the Boost binaries,
e.g. C:\Program Files\boost\boost_1_55_0\lib\., C:\boost_1_61_0\stage\lib
From the Build menu, select Build Solution.
*/

#include "boost/multi_array.hpp"
#include <cassert>

int main_ka() {
	// Create a 3D array that is 3 x 4 x 2
	typedef boost::multi_array<double, 3> array_type;
	typedef array_type::index index;
	array_type A(boost::extents[3][4][2]);

	// Assign values to the elements
	int values = 0;
	for (index i = 0; i != 3; ++i)
		for (index j = 0; j != 4; ++j)
			for (index k = 0; k != 2; ++k)
				A[i][j][k] = values++;

	// Verify values
	int verify = 0;
	for (index i = 0; i != 3; ++i)
		for (index j = 0; j != 4; ++j)
			for (index k = 0; k != 2; ++k)
				assert(A[i][j][k] == verify++);

	return 0;
}