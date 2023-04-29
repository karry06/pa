#include <pthread.h>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include<arm_neon.h>
using namespace std;
const int N=2362;//����Ĺ�ģ
char subMatrix[N][N+1]; //��Ԫ��
char eliminatedRows[N][N]; //����Ԫ��
int bLength; //����Ԫ�г���
int NUM_THREADS=7;
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
}}

struct ThreadData {
    int start;
    int end;
};

void* SGE_thread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    for(int i=data->start; i<=data->end; i++) {
        for(int j=N-1; j>=0; j--) {
            if(eliminatedRows[i][j] == 1) {
                if(subMatrix[j][N] != 0) {
                    for(int k=0; k<=j; k++) {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                } else {
                    for(int k=0; k<=j; k++) {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1;
                    break;
                }
            }
        }
    }
    delete data;
    pthread_exit(NULL);
}

void SGE_parallel() {
    pthread_t threads[NUM_THREADS];
    int rows_per_thread = bLength / NUM_THREADS;
    for(int i=0; i<NUM_THREADS; i++) {
        ThreadData* data = new ThreadData();
        data->start = i * rows_per_thread;
        data->end = (i == NUM_THREADS-1) ? bLength-1 : data->start + rows_per_thread - 1;
        pthread_create(&threads[i], NULL, SGE_thread, data);
    }
    for(int i=0; i<NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
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
    //SGE_Pthread
    gettimeofday(&beg1,NULL);
    SGE_parallel();
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"SGE_Pthread is "<< time/1000<<" ms"<<endl;

}
