#include <matrix.h>

Matrix::Matrix()
{
    if(!empty())
    {
        clear();
    }
	n_rows = 0;
	n_cols = 0;
	data = NULL;
}

bool Matrix::empty()const
{
    return (n_rows == 0 && n_cols == 0);
}

Matrix::Matrix(int rows, int cols)
{
    if(!empty())
    {
        clear();
    }
	n_rows = rows;
	n_cols = cols;
	data   = new double*[rows];
	for(int i = 0; i < rows; i++)
	{
		data[i] = new double[cols];
		for(int j = 0; j < cols; j++)
		{
			data[i][j] = 0;
		}
	}
}

Matrix::Matrix(const Matrix& A)
{
    if(!empty())
    {
        clear();
    }

    n_rows = A.n_rows;
    n_cols = A.n_cols;

    data = new double*[n_rows];
    for(int i = 0; i < n_rows; i++)
	{
        data[i] = new double[n_cols];
        for(int j = 0; j < n_cols; j++)
		{
			data[i][j] = A.data[i][j];
		}
	}
}

Matrix::~Matrix()
{
	for(int i = 0; i < n_rows; i++)
	{
		delete [] data[i];
	}
    if(data)
    {
        delete [] data;
    }
	data = NULL;
	n_rows = 0;
	n_cols = 0;
}

void Matrix::clear()
{
	this->~Matrix();
}

Matrix Matrix::input()
{
	int rows, cols;
	if(empty())
	{
		cout << "Input the rows of your matrix:";
		cin >> rows;
		cout << "Input the cols of your matrix:";
		cin >> cols;
		*this = Matrix(rows, cols);
		cout << "Input the data of your matrix:" << endl;
	}
	
	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_cols; j++)
		{
			cin >> data[i][j];
		}
	}
	cout << "Input end!" << endl;

	return *this;
}

Matrix& Matrix::operator =(const Matrix& A)
{
    if(!empty())
    {
        clear();
    }

	n_rows = A.n_rows;
	n_cols = A.n_cols;

	data = new double*[n_rows];
	for(int i = 0; i < n_rows; i++)
	{
		data[i] = new double[n_cols];
		for(int j = 0; j < n_cols; j++)
		{
			data[i][j] = A.data[i][j];
		}
	}

	return *this;
}

Matrix Matrix::operator -()const
{
	Matrix B(n_rows, n_cols);
	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_cols; j++)
		{
			B.data[i][j] = -data[i][j];
		}
	}

	return B;
}

Matrix operator +(const Matrix& A, const Matrix& B)
{
	if(A.n_rows != B.n_rows || A.n_cols != B.n_cols)
	{
		cout << "[ Error in \"Matrix A + Matrix B\":" << endl
			 << "  A.n_rows != B.n_rows || A.n_cols != B.n_cols is not permitted! ]" << endl;
		exit(-1);
	}

	Matrix C(A.n_rows, A.n_cols);
	for(int i = 0; i < A.n_rows; i++)
	{
		for(int j = 0; j < A.n_cols; j++)
		{
			C.data[i][j] = A.data[i][j] + B.data[i][j];
		}
	}

	return C;
}

Matrix operator -(const Matrix& A, const Matrix& B)
{
	if(A.n_rows != B.n_rows || A.n_cols != B.n_cols)
	{
		cout << "[ Error in \"Matrix A - Matrix B\":" << endl
			 << "  A.n_rows != B.n_rows || A.n_cols != B.n_cols is not permitted! ]" << endl;
		exit(-1);
	}

	Matrix C(A.n_rows, A.n_cols);
	for(int i = 0; i < A.n_rows; i++)
	{
		for(int j = 0; j < A.n_cols; j++)
		{
			C.data[i][j] = A.data[i][j] - B.data[i][j];
		}
	}

	return C;
}

Matrix operator *(const Matrix& A, const Matrix& B)
{
	if(A.n_cols != B.n_rows)
	{
		cout << "[ Error in \"Matrix A * Matrix B\":" << endl
			 << "  A.n_cols != B.n_rows is not permitted!" << endl;
		exit(-1);
	}

	Matrix C(A.n_rows, B.n_cols);
	for(int i = 0; i < A.n_rows; i++)
	{
		for(int j = 0; j < B.n_cols; j++)
		{
			double S = 0;
			for(int k = 0; k < A.n_cols; k++)
			{
				S += A.data[i][k] * B.data[k][j];
			}
			C.data[i][j] = S;
		}
	}

	return C;
}

Matrix operator *(double scale, const Matrix& A)
{
	Matrix B(A.n_rows, A.n_cols);
	for(int i = 0; i < B.n_rows; i++)
	{
		for(int j = 0; j < B.n_cols; j++)
		{
			B.data[i][j] = scale * A.data[i][j];
		}
	}

	return B;
}

Matrix operator *(const Matrix& A, double scale)
{
	return scale * A;
}

Matrix operator /(const Matrix& A, double scale)
{
	Matrix B(A.n_rows, A.n_cols);
	for(int i = 0; i < B.n_rows; i++)
	{
		for(int j = 0; j < B.n_cols; j++)
		{
			B.data[i][j] = A.data[i][j] / scale;
		}
	}

	return B;
}

bool operator ==(const Matrix& A, const Matrix& B)
{
	if(A.n_rows != B.n_rows || A.n_cols != B.n_cols)
	{
		return false;
	}

	for(int i = 0; i < A.n_rows; i++)
	{
		for(int j = 0; j < A.n_cols; j++)
		{
			if(A.data[i][j] != B.data[i][j])
			{
				return false;
			}
		}
	}

	return true;
}

bool operator !=(const Matrix& A, const Matrix& B)
{
	return !(A == B);
}

Matrix Matrix::t()const
{
	Matrix B(n_cols, n_rows);
	for(int i = 0; i < B.n_rows; i++)
	{
		for(int j = 0; j < B.n_cols; j++)
		{
			B.data[i][j] = data[j][i];
		}
	}

	return B;
}

double Matrix::trac()const
{
	if(n_rows != n_cols)
	{
		cout << "[ Error in \"double Matrix::trac()const\":" << endl
			 << "  rows != cols is not permitted!" << endl;
		exit(-1);
	}

	double T = 0;
	for(int i = 0; i < n_rows; i++)
	{
		T += data[i][i];
	}

	return T;
}

Matrix eye(const int& n)
{
	Matrix In(n, n);

	for(int i = 0; i < n; i++)
	{
		In.data[i][i] = 1;
	}

	return In;
}

Matrix zeros(const int& rows, const int& cols)
{
	return Matrix(rows, cols);
}

Matrix ones(const int& rows, const int& cols)
{
	Matrix A(rows, cols);
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			A.data[i][j] = 1;
		}
	}

	return A;
}

string multi_space(const int& n)
{
	string space;
	for(int i = 0; i < n; i++)
	{
		space += " ";
	}

	return space;
}

string num2str(const double& x)
{
	stringstream ss;
	string str;
	ss << x;
	ss >> str;
	return str;
}

ostream & operator <<(ostream& o, Matrix A)
{
	correct(A);
	o << endl;
	int *longest_size = new int[A.n_cols];
	for(int j = 0; j < A.n_cols; j++)
	{
		longest_size[j] = num2str(A.data[0][j]).size();
		for(int i = 1; i < A.n_rows; i++)
		{
			int current_size = num2str(A.data[i][j]).size();
			if(current_size > longest_size[j])
			{
				longest_size[j] = current_size;
			}
		}
	}

	for(int i = 0; i < A.n_rows; i++)
	{
		for(int j = 0; j < A.n_cols; j++)
		{
			int space_length = longest_size[j] - num2str(A.data[i][j]).size() + 2;
			if(A.data[i][j] >= 0)
			{
				o << " " << A.data[i][j] << multi_space(space_length);
			}
			else
			{
				o << A.data[i][j] << multi_space(space_length+1);
			}
		}
		o << endl;
	}

	return o;
}

istream& operator >>(istream& i, Matrix& A)
{
	A.input();
	return i;
}

Matrix Matrix::exchange_row(const int& row1, const int& row2)
{
	for(int j = 0; j < n_cols; j++)
	{
		swap(data[row1][j], data[row2][j]);
	}

	return *this;
}

Matrix Matrix::scale_row(const double& scale, const int& row)
{
	for(int j = 0; j < n_cols; j++)
	{
		data[row][j] *= scale;
	}

	return *this;
}

Matrix Matrix::scale_add_row(const int& row1, const double& scale, const int& row2)
{
	for(int j = 0; j < n_cols; j++)
	{
		data[row2][j] += scale * data[row1][j];
	}

	return *this;
}

Matrix Matrix::exchange_col(const int& col1, const int& col2)
{
	for(int i = 0; i < n_rows; i++)
	{
		swap(data[i][col1], data[i][col2]);
	}

	return *this;
}

Matrix Matrix::scale_col(const double& scale, const int& col)
{
	for(int i = 0; i < n_rows; i++)
	{
		data[i][col] *= scale;
	}

	return *this;
}

Matrix Matrix::scale_add_col(const int& col1, const double& scale, const int& col2)
{
	for(int i = 0; i < n_rows; i++)
	{
		data[i][col2] += scale * data[i][col1];
	}

	return *this;
}

int it_row_max(Matrix A, int it_row, int it_col)
{
	double row_max = A.data[it_row][it_col];
	int it_Row_Max = it_row;
	for(int row = it_row + 1; row < A.n_rows; row++)
	{
		if(fabs(A.data[row][it_col]) > fabs(row_max) )
		{
			row_max = A.data[row][it_col];
			it_Row_Max = row;
		}
	}
	return it_Row_Max;
}

Matrix mirror(const Matrix& A)
{
	int rows = A.n_rows;
	int cols = A.n_cols;

	Matrix B(rows, cols);
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			B.data[i][j] = A.data[rows-1-i][cols-1-j];
		}
	}

	return B;
}

Matrix Matrix::reduce()const
{
	Matrix Result = *this;

	for(int it_row = 0; it_row < n_rows; it_row++)
	{
		int it_col_nonzero = -1;
		for(int it_col = it_row; it_col < n_cols; it_col++)
		{
			for(int subit_row = it_row; subit_row < n_rows; subit_row++)
			{
				if( !isZero(Result.data[subit_row][it_col]) )
				{
					it_col_nonzero = it_col;
					break;
				}
			}
			if(it_col_nonzero != -1)
			{
				break;
			}
		}

		if(it_col_nonzero == -1)
		{
			break;
		}

		int row_main = it_row_max(Result, it_row, it_col_nonzero);
		if(it_row != row_main)
		{
			Result.exchange_row(it_row, row_main);
		}
		
		Result.scale_row(1.0/Result.data[it_row][it_col_nonzero], it_row);
	
		for(int subit_row = 0; subit_row < n_rows; subit_row++)
		{
			if(subit_row != it_row)
			{
				double scale = - Result.data[subit_row][it_col_nonzero] / Result.data[it_row][it_col_nonzero];
				Result.scale_add_row(it_row, scale, subit_row);
			}
		}
	}
	return Result;
}

Matrix Matrix::inv()const
{
	if(n_rows != n_cols)
	{
		cout << "[ Error in \"Matrix Matrix::inv()const\":" << endl
			 << "  rows != cols is not permitted! ]" << endl;
		exit(-1);
	}
	if(det() == 0)
	{
		cout << "[ Error in \"Matrix Matrix::inv()const\":" << endl
			 << "  det(A) = 0, then, A has no inverse! ]" << endl;
		exit(-1);
	}

	Matrix Temp(n_rows, 2 * n_rows);
	for(int i = 0; i < n_rows; i++)
	{
		int j;
		for(j = 0; j < n_rows; j++)
		{
			Temp.data[i][j] = data[i][j];
		}
		Temp.data[i][n_rows+i] = 1;
	}

	Temp = Temp.reduce();

	Matrix Result(n_rows, n_rows);
	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_rows; j++)
		{
			Result.data[i][j] = Temp.data[i][n_rows+j];
		}
	}

	return Result;
}

double Matrix::det()const
{
	if(n_rows != n_cols)
	{
		cout << "[ Error in \"double Matrix::det()const\":" << endl
			 << "  n_rows != n_cols is not permitted!" << endl;
		exit(-1);
	}

	Matrix Result = *this;
	int times_change = 0;
	for(int it_row = 0; it_row < n_rows-1; it_row++)
	{
		int it_col_nonzero = -1;
		for(int it_col = it_row; it_col < n_rows; it_col++)
		{
			for(int subit_row = it_row; subit_row < n_rows; subit_row++)
			{
				if(Result.data[subit_row][it_col] != 0)
				{
					it_col_nonzero = it_col;
					break;
				}
			}
			if(it_col_nonzero >= 0)
			{
				break;
			}
		}

		int row_main = it_row_max(Result, it_row, it_col_nonzero);
		if(it_row != row_main)
		{
			times_change++;
			Result.exchange_row(it_row, row_main);
		}
		
		for(int subit_row = it_row + 1; subit_row < n_rows; subit_row++)
		{
			double scale = - Result.data[subit_row][it_col_nonzero] / Result.data[it_row][it_col_nonzero];
			Result.scale_add_row(it_row, scale, subit_row);
		}
	}

	double Det = 1;
	for(int i = 0; i < n_rows; i++)
	{
		Det *= Result.data[i][i];
	}

	int sgn = 1;
	for(int i = 1; i <= times_change; i++)
	{
		sgn = -sgn;
	}

	return sgn * Det;
}

Matrix correct(Matrix& A)
{
	for(int i = 0; i < A.n_rows; i++)
	{
		for(int j = 0; j < A.n_cols; j++)
		{
			if(A.data[i][j] > -1E-6 && A.data[i][j] < 1E-6)
			{
				A.data[i][j] = 0;
			}
		}
	}
	return A;
}

int Matrix::rank()const
{
	Matrix Result = *this;
	int times_change = 0;
	for(int it_row = 0; it_row < n_rows-1; it_row++)
	{
		int it_col_nonzero = -1;
		for(int it_col = it_row; it_col < n_cols; it_col++)
		{
			for(int subit_row = it_row; subit_row < n_rows; subit_row++)
			{
				if(Result.data[subit_row][it_col] != 0)
				{
					it_col_nonzero = it_col;
					break;
				}
			}
			if(it_col_nonzero >= 0)
			{
				break;
			}
		}

		int row_main = it_row_max(Result, it_row, it_col_nonzero);
		if(it_row != row_main)
		{
			times_change++;
			Result.exchange_row(it_row, row_main);
		}
		
		for(int subit_row = it_row + 1; subit_row < n_rows; subit_row++)
		{
			double scale = - Result.data[subit_row][it_col_nonzero] / Result.data[it_row][it_col_nonzero];
			Result.scale_add_row(it_row, scale, subit_row);
		}
	}

	Matrix Test(n_rows + 1, n_cols + 1);
	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 1; j < n_cols + 1; j++)
		{
			Test.data[i][j] = Result.data[i][j - 1];
		}
	}

	correct(Test);

	int Rank = 0;
	for(int col = 1; col < n_cols + 1; col++)
	{
		for(int row = 0; row < n_rows; row++)
		{
			if(Test.data[row][col] != 0 && Test.data[row][col-1] == 0 && Test.data[row + 1][col] == 0)
			{
				Rank++;
			}
		}
	}

	return Rank;
}

Matrix Matrix::col(const int& n)const
{
	if(n < 0 || n >= n_cols)
	{
		cout << "[ Error in \"Matrix Matrix::col(int n)\":" << endl
			 << "  n is not in area! ]" << endl;
		exit(-1);
	}

	Matrix COL_n(n_rows, 1);
	for(int it_row = 0; it_row < n_rows; it_row++)
	{
		COL_n.data[it_row][0] = data[it_row][n];
	}

	return COL_n;
}

Matrix Matrix::cat(const Matrix& A)
{
	if(n_rows != A.n_rows)
	{
		cout << "[ Error in \"Matrix Matrix::cat(const Matrix& A)\":" << endl
			 << "  rows must be equal! ]" << endl;
		exit(-1);
	}

	Matrix Result(n_rows, n_cols + A.n_cols);
	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_cols; j++)
		{
			Result.data[i][j] = data[i][j];
		}
		for(int j = n_cols; j < Result.n_cols; j++)
		{
			Result.data[i][j] = A.data[i][j-n_cols];
		}
	}

	*this = Result;

	return *this;
}

double norm(const Matrix& A)
{
	if(A.n_cols != 1)
	{
		cout << "[ Error in \"double norm(const Matrix& A)\":" << endl
			 << "  cols != 1 is not permitted! ]" << endl;
		exit(-1); 
	}

	return sqrt( (A.t() * A).data[0][0] );
}

Matrix normalize(const Matrix& A)
{
	if(A.n_cols != 1)
	{
		cout << "[ Error in \"double norm(const Matrix& A)\":" << endl
			 << "  cols != 1 is not permitted! ]" << endl;
		exit(-1); 
	}

	double N = sqrt( (A.t() * A).data[0][0] );
	if(!isZero(N))
	{
		return A / N;
	}
	else
	{
		return Matrix(A.n_rows, 1);
	}
}

Matrix Gram_Schmidt(const Matrix& A)
{
	Matrix Bases = normalize(A.col(0));

	for(int it_col = 1; it_col < A.n_cols; it_col++)
	{
		Matrix vk = A.col(it_col);
		Matrix S(A.n_rows, 1);
		for(int i = 0; i < it_col; i++)
		{
			Matrix ei = Bases.col(i);
			double num = (ei.t() * vk).data[0][0];
			double den = (ei.t() * ei).data[0][0];
			if(!isZero(den))
			{
				S = S + num / den * ei;
			}
		}
		Bases.cat( normalize(vk - S) );
		S.clear();
	}

	return Bases;
}

Matrix orthonormalize(Matrix A)
{
	Matrix H = eye(A.n_rows);

	for(int j = 1; j < A.n_cols; j++)
	{
		int n = A.n_rows - j + 1;
	    Matrix e(n, 1);
	    e.data[0][0] = 1;

	    Matrix v(n, 1);
	    for(int i = 0; i < n; i++)
	    {
	    	v.data[i][0] = A.data[i+j-1][j-1];
	    }

	    Matrix u = v - norm(v) * e;
	    u = normalize(u);

	    Matrix h = eye(n) - 2 * u * u.t();   

	    Matrix K = eye(A.n_rows);
	    for(int sub_i = j-1; sub_i < A.n_rows; sub_i++)
	    {
	    	for(int sub_j = j-1; sub_j < A.n_cols; sub_j++)
	    	{
	    		K.data[sub_i][sub_j] = h.data[sub_i-j+1][sub_j-j+1];
	    	}
	    }

	    H = K * H;
	    A = H * A;
	}

	return H.t();
}

vector<Matrix> Matrix::QR()const
{
    static int times_qr = 0;
    cout << times_qr++;
	Matrix Q = orthonormalize(*this);

	Matrix R = Q.t() * (*this);
	correct(R);

	vector<Matrix> qr;
	qr.push_back(Q);
	qr.push_back(R);
	return qr;
}

Matrix Householder(const Matrix& A_init)
{
	Matrix A = A_init;
	int rows = A.n_rows;
	int cols = A.n_cols;

	if(rows != cols)
	{
		cout << "[ Error in \"Matrix Householder(const Matrix& A)\":" << endl
			 << "  rows != cols is not permitted! ]" << endl;
		exit(-1); 
	}

	int n = rows;
	for(int i = 1; i <= n-2; i++)
	{
		Matrix e(n-i, 1);
		Matrix x(n-i, 1);
		e.data[0][0] = 1;
		for(int it_row = 0; it_row < n-i; it_row++)
		{
			x.data[it_row][0] = A.data[i+it_row][i-1];
		}
		double alpha = norm(x);
		Matrix Omega = x - alpha*e;
		Omega = Omega / norm(Omega);

		Matrix H0 = eye(n-i) - 2 * Omega * Omega.t();
		Matrix H  = eye(n);

		for(int it_row = i; it_row < n; it_row++)
		{
			for(int it_col = i; it_col < n; it_col++)
			{
				H.data[it_row][it_col] = H0.data[it_row-i][it_col-i];
			}
		}

		A = H * A * H;
	}

	return A;
}

double max(vector<double> X)
{
	vector<double>::iterator it_Max = X.begin();
	for(vector<double>::iterator it = X.begin(); it != X.end(); it++)
	{
		if(*it > *it_Max)
		{
			it_Max = it;
		}
	}
	return *it_Max;
}

Matrix extractMat(const Matrix& A, int it_col)
{
	Matrix Result(2, 2);

	Result.data[0][0] = A.data[it_col][it_col];
	Result.data[0][1] = A.data[it_col][it_col+1];
	Result.data[1][0] = A.data[it_col+1][it_col];
	Result.data[1][1] = A.data[it_col+1][it_col+1];

	return Result;
}

bool conver(const Matrix& A, double epsillon)
{
	int rows = A.n_rows;
	int cols = A.n_cols;
	if(rows != cols)
	{
		cout << "[ Error in \"bool conver(const Matrix& A)\":" << endl
			 << "  A.n_rows != A.n_cols is not permitted! ]" << endl;
		exit(-1);
	}

	int n = rows;

	if(n < 3)
	{
		return true;
	}

	for(int i = 2; i < n; i++)
	{
		for(int j = 0; j <= i-2; j++)
		{
			if( fabs(A.data[i][j]) > epsillon )
			{
				return false;
			}
		}
	}

	for(int i = 1; i < n-1; i++)
	{
		if( fabs(A.data[i][i-1]) > epsillon && fabs(A.data[i+1][i]) > epsillon )
		{
			return false;
		}
	}

	return true;
}

void sort_norm(vector<Complex>& X)
{
    int n = X.size();
    for(int i = n-2; i >= 0; i--)
    {
        for(int j = 0; j <= i; j++)
        {
            if(X[j].module() < X[j+1].module())
            {
                Complex temp = X[j+1];
                X[j+1] = X[j];
                X[j] = temp;
            }
        }
    }
}

vector<Complex> Matrix::eigenvalue()const
{
	if(n_rows != n_cols)
	{
		cout << "[ Error in \"vector<Complex> Matrix::eigenvalue()const\":" << endl
			 << "  rows != cols is not permitted! ]" << endl;
		exit(-1); 
	}

	vector<Complex> Eigenvalue;

	if(n_rows == 2)
	{
		Complex b = -trac();
		Complex c =  det();

		Complex lambda1 = ( -b + pow( pow(b, 2) - 4 * c, 0.5 ) ) / 2;
		Complex lambda2 = ( -b - pow( pow(b, 2) - 4 * c, 0.5 ) ) / 2;

		Eigenvalue.push_back(lambda1);
		Eigenvalue.push_back(lambda2);

		return Eigenvalue;
	}

	Matrix A = *this;

	int times = 0;
	double epsillon = 1E-7;
    Matrix In = eye(n_rows);
    while( !conver(A, epsillon) )
    {
		double u = A.data[n_rows-1][n_rows-1];
        A = A - u * In;

        Matrix Q = orthonormalize(A);
        Matrix R = Q.t() * A;
        correct(R);
        A = R * Q + u * In;

        times++;
	}

	for(int j = 0; j < n_rows-1; j++)
	{
		if( fabs(A.data[j+1][j]) > epsillon )
		{
			Matrix subMat = extractMat(A, j);
			vector<Complex> Lambda = subMat.eigenvalue();
			Eigenvalue.push_back(Lambda[0]);
			Eigenvalue.push_back(Lambda[1]);
			Lambda.clear();
		}
		else if( fabs(A.data[j][j-1]) <= epsillon )
		{
            Eigenvalue.push_back( Complex(A.data[j][j]) );
		}	
	}

	if( fabs(A.data[n_rows-1][n_rows-2]) < epsillon )
	{
        Eigenvalue.push_back( Complex(A.data[n_rows-1][n_rows-1]) );
	}
    sort_norm(Eigenvalue);
	return Eigenvalue;
}


Matrix Matrix::eigenvector(const double& lambda)const
{
	if(n_rows != n_cols)
	{
		cout << "[ Error in \"Matrix Matrix::eigenvector(const double&)const\" ]" << endl
			 << "[ current matrix is not a square matrix!                      ]" << endl;
		exit(-1);
	}

	Matrix B = *this - lambda * eye(n_rows);
//	if( !isZero(B.det()) )
//	{
//		cout << "The input number is not a eigenvalue of current matrix!" << endl;
//		exit(-1);
//	}

	B = B.reduce();
	Matrix v = B.col(n_cols-1);
	v.data[n_rows-1][0] = -1;
	return v;
}
