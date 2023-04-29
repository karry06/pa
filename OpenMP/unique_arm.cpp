#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include <omp.h>
using namespace std;
const int N=2362;//����Ĺ�ģ
char subMatrix[N][N+1]; //��Ԫ��
char eliminatedRows[N][N]; //����Ԫ��
int bLength; //����Ԫ�г���
void creatSubMatrix(char const *fname)
{
    // ������Ԫ�Ӿ���
    ifstream fin;
    fin.open(fname, ios::in);
    if (!fin.is_open()) // ���ļ�ʧ��
    {
        cout << "Failed to open file." << endl;
        exit(0);
    }
    else // ���ļ��ɹ�
    {
        string strTemp;
        while (getline(fin, strTemp)) // ��ȡ�ļ���ÿһ��
        {
            int row_idx, col_idx,first=1;
            istringstream isTemp(strTemp); // ��������ַ���ת��Ϊ������
            while (isTemp >> col_idx) // ����������������
            {
                if(first)//����ǵ�һ��
                {
                    row_idx=col_idx;//��һ��Ϊ������
                    subMatrix[row_idx][N] = 1;//�ھ����б�Ǹ��е�ĩβλ��Ϊ 1
                    first=0;
                }
                subMatrix[row_idx][col_idx] = 1; // ����Ԫ�Ӿ����б�Ǹ�Ԫ��
            }

        }
        fin.close(); // �ر��ļ�
    }
}

void creatbxyhMatrix(char const *fname)
{
    //��������Ԫ�о���
    ifstream fin;
    fin.open(fname, ios::in);
    if (!fin.is_open())
    {
        cout << "Failed to open file." << endl;
        exit(0);
    }
    else
    {
        string strTemp;
        while (!fin.eof())
        {
            int col_idx;
            getline(fin, strTemp);// ��ȡ�ļ���ÿһ��
            istringstream isTemp(strTemp); // ��������ַ���ת��Ϊ������
            while (isTemp >> col_idx) // �����������е�������
            {
                eliminatedRows[bLength][col_idx] = 1; // �ڱ���Ԫ�о����б�Ǹ�Ԫ��
            }
            bLength++;
        }
        bLength--; // ���һ�п����ǿ��У������Ҫ��ȥһ��
        fin.close(); // �ر��ļ�

    }
}

void SGE() //�����˹��ȥ
{
    for(int i=0; i<bLength; i++) // ��ÿ������Ԫ�н��д���
    {
        for(int j=N-1; j>=0; j--) // �����һ�п�ʼ��ǰ����
        {
            if(eliminatedRows[i][j]==1);  // ���void SGE() //�����˹��ȥ
		}
    for(int i=0; i<bLength; i++) // ��ÿ������Ԫ�н��д���
    {
        for(int j=N-1; j>=0; j--) // �����һ�п�ʼ��ǰ����
        {
            if(eliminatedRows[i][j]==1)  // ���eliminatedRows[i][j]Ϊ1��˵��������Ҫ������Ԫ
            {
                if(subMatrix[j][N]!=0) // ���subMatrix[j][N]!=0��˵�����д�����Ԫ���У���Ҫִ��������������Ԫ
                {
                    // ��ÿ��Ԫ��ִ��������������eliminatedRows[i][k]
                    for(int k=0; k<=j; k++)
                    {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                }
                else // ���subMatrix[j][N]==0��˵������û����Ԫ���У���eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                {
                    // ��eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                    for(int k=0; k<=j; k++)
                    {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // ��Ǹ����Ѿ�������
                    break; // ����ѭ��������������һ������Ԫ��
                }
            }
        }
    }
	}
}
void SGE1()
{
    #pragma omp parallel for schedule(dynamic) shared(eliminatedRows, subMatrix, bLength)
    for(int i=0; i<bLength; i++) // ��ÿ������Ԫ�н��д���
    {
        for(int j=N-1; j>=0; j--) // �����һ�п�ʼ��ǰ����
        {
            if(eliminatedRows[i][j]==1)  // ���eliminatedRows[i][j]Ϊ1��˵��������Ҫ������Ԫ
            {
                if(subMatrix[j][N]!=0) // ���subMatrix[j][N]!=0��˵�����д�����Ԫ���У���Ҫִ��������������Ԫ
                {
                    // ��ÿ��Ԫ��ִ��������������eliminatedRows[i][k]
                    //#pragma omp atomic
                    #pragma omp critical
                    for(int k=0; k<=j; k++)
                    {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                }
                else // ���subMatrix[j][N]==0��˵������û����Ԫ���У���eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                {
                    // ��eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                    for(int k=0; k<=j; k++)
                    {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // ��Ǹ����Ѿ�������
                    break; // ����ѭ��������������һ������Ԫ��
                }
            }
        }
    }
}
int main()
{
	creatSubMatrix("/home/data/Groebner/5_2362_1226_453/1.txt");
	creatbxyhMatrix("/home/data/Groebner/5_2362_1226_453/2.txt");
	srand(time(0));
    struct timeval beg1,end1,beg2,end2;
    float time;
    //SGE
    gettimeofday(&beg1,NULL);
    SGE();
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"SGE is "<< time/1000<<" ms"<<endl;
    //SGE_OpenMP
    gettimeofday(&beg1,NULL);
    SGE1();
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"SGE_OpenMP is "<< time/1000<<" ms"<<endl;

}

