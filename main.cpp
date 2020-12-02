#include <matrix.hpp>

int main()
{
	Matrix<float> A = rand(3, 3);
	Matrix<float> B = {{1,2,3},
					 {3,8,3},
					 {2,5,6}};
	cout << A + Matrix<float>(B) << endl;
	return 0;
}