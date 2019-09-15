#include <matrix.h>

int main()
{
	Matrix<float> A = rand(3, 3);
	Matrix<int> B = {{1,2,3},
					 {3,8,3},
					 {2,5,6}};
	cout << A + Matrix<float>(B) << endl;
	return 0;
}