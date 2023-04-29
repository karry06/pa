#include <pthread.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <semaphore.h>
#include <unistd.h>
#include <omp.h>
using namespace std;
#define NUM_THREADS1 8
#define NUM_THREADS2 6
#define NUM_THREADS3 4
#define CHUNK_SIZE 8
void init(float** a,int n) {//������������
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            a[i][j] = 0;
        }
        a[i][i] = 1.0;
        for(int j = i + 1; j < n; j++) {
            a[i][j] = rand();
        }
    }
    for(int k = 0; k < n; k++) {
        for(int i = k + 1; i < n; i++) {
            for(int j = 0; j < n; j++) {
                a[i][j] += a[k][j];
            }
        }
    }
}
void Trivial(float** a, int n) {  // ƽ����˹��ȥ�㷨
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j ++)
		{
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
void OpenMP_Static8(float** a, int n)//ʹ�þ�̬���ȣ�8���߳�
{
    #pragma omp parallel num_threads(NUM_THREADS1) shared(a,n)//ʹ�� 8 ���̣߳� a �� n �ǹ���ı��������Ա�����߳�ͬʱ����
    {
        // ��˹��Ԫ�ĵ�һ��ѭ��
        for (int k = 0; k < n; k++)
        {
            // ʹ�þ�̬���Ȳ��м���
            #pragma omp for schedule(static)
            for (int j = k + 1; j < n; j++)
                a[k][j] = a[k][j] / a[k][k];
            // �������̼߳����� a[k][j] �󣬵ȴ������߳�
            #pragma omp barrier
            // ��һ���߳̽� a[k][k] ����Ϊ 1
            #pragma omp single
            a[k][k] = 1.0;
            // �������̼߳����� a[k][k] �󣬵ȴ������߳�
            #pragma omp barrier
            // Ƕ��ѭ��չ��Ϊһ��ѭ������̬����
            #pragma omp for collapse(2) schedule(static)
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    a[i][j] = a[i][j] - a[i][k] * a[k][j];
            // �������̼߳����� a[i][j] �󣬵ȴ������߳�
            #pragma omp barrier
            // ʹ�þ�̬���Ȳ��м��� a[i][k]
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++)
                a[i][k] = 0;
        }
    }
}
void OpenMP_Static6(float** a, int n)//ʹ�þ�̬���ȣ�6���߳�
{
    #pragma omp parallel num_threads(NUM_THREADS2) shared(a,n)
    {
        for (int k = 0; k < n; k++)
        {
            #pragma omp for schedule(static)
            for (int j = k + 1; j < n; j++)
                a[k][j] = a[k][j] / a[k][k];
            #pragma omp barrier
            #pragma omp single
            a[k][k] = 1.0;
            #pragma omp barrier
            #pragma omp for collapse(2) schedule(static)
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    a[i][j] = a[i][j] - a[i][k] * a[k][j];
            #pragma omp barrier
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++)
                a[i][k] = 0;
        }
    }
}
void OpenMP_Static4(float** a, int n)//ʹ�þ�̬���ȣ�4���߳�
{
    #pragma omp parallel num_threads(NUM_THREADS3) shared(a,n)
    {
        for (int k = 0; k < n; k++)
        {
            #pragma omp for schedule(static)
            for (int j = k + 1; j < n; j++)
                a[k][j] = a[k][j] / a[k][k];
            #pragma omp barrier
            #pragma omp single
            a[k][k] = 1.0;
            #pragma omp barrier
            #pragma omp for collapse(2) schedule(static)
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    a[i][j] = a[i][j] - a[i][k] * a[k][j];
            #pragma omp barrier
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++)
                a[i][k] = 0;
        }
    }
}

void OpenMP_Dynamic8(float** a, int n)//ʹ�ö�̬���ȣ�8���߳�
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS1) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
            a[k][j] = a[k][j] / a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}
void OpenMP_Dynamic6(float** a, int n)//ʹ�ö�̬���ȣ�6���߳�
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS2) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
            a[k][j] = a[k][j] / a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}
void OpenMP_Dynamic4(float** a, int n)//ʹ�ö�̬���ȣ�4���߳�
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS3) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
            a[k][j] = a[k][j] / a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}
void OpenMP_Guided8(float** a, int n)//ʹ�õ�����ȣ�8���߳�
{
    #pragma omp parallel num_threads(NUM_THREADS1) shared(a,n)
    {
        for (int k = 0; k < n; k++)
        {
            #pragma omp for schedule(guided)
            for (int j = k + 1; j < n; j++)
                a[k][j] = a[k][j] / a[k][k];
            #pragma omp barrier
            #pragma omp critical
            a[k][k] = 1.0;
            #pragma omp barrier
            #pragma omp for collapse(2) schedule(guided)
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    a[i][j] = a[i][j] - a[i][k] * a[k][j];
            #pragma omp barrier
            #pragma omp for schedule(guided)
            for (int i = k + 1; i < n; i++)
                a[i][k] = 0;
        }
    }
}
void OpenMP_Guided6(float** a, int n)//ʹ�õ�����ȣ�6���߳�
{
    #pragma omp parallel num_threads(NUM_THREADS2) shared(a,n)
    {
        for (int k = 0; k < n; k++)
        {
            #pragma omp for schedule(guided)
            for (int j = k + 1; j < n; j++)
                a[k][j] = a[k][j] / a[k][k];
            #pragma omp barrier
            #pragma omp critical
            a[k][k] = 1.0;
            #pragma omp barrier
            #pragma omp for collapse(2) schedule(guided)
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    a[i][j] = a[i][j] - a[i][k] * a[k][j];
            #pragma omp barrier
            #pragma omp for schedule(guided)
            for (int i = k + 1; i < n; i++)
                a[i][k] = 0;
        }
    }
}
void OpenMP_Guided4(float** a, int n)//ʹ�õ�����ȣ�4���߳�
{
    #pragma omp parallel num_threads(NUM_THREADS3) shared(a,n)
    {
        for (int k = 0; k < n; k++)
        {
            #pragma omp for schedule(guided)
            for (int j = k + 1; j < n; j++)
                a[k][j] = a[k][j] / a[k][k];
            #pragma omp barrier
            #pragma omp critical
            a[k][k] = 1.0;
            #pragma omp barrier
            #pragma omp for collapse(2) schedule(guided)
            for (int i = k + 1; i < n; i++)
                for (int j = k + 1; j < n; j++)
                    a[i][j] = a[i][j] - a[i][k] * a[k][j];
            #pragma omp barrier
            #pragma omp for schedule(guided)
            for (int i = k + 1; i < n; i++)
                a[i][k] = 0;
        }
    }
}
int main()
{
    int n[8]={64,128,256,512,1024,1536,2048,3000};
    for(int i=0;i<8;i++)
    {
        float** test = new float*[n[i]];
        for (int j = 0; j < n[i]; j++)
        {
            test[j] = new float[n[i]];
        }
        init(test,n[i]);
        cout<<"N is:"<<n[i]<<endl;
        struct timeval beg1,end1;
		srand(time(0)); 
		float time;
	init(test,n[i]);
    gettimeofday(&beg1,NULL);
    Trivial(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"Normal:"<< time/1000<<" ms"<<endl;

    init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Static8(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Static8:"<< time/1000<<" ms"<<endl;
	
	init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Static6(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Static6:"<< time/1000<<" ms"<<endl;
    
    init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Static4(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Static4:"<< time/1000<<" ms"<<endl;
    
    init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Dynamic8(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Dynamic8:"<< time/1000<<" ms"<<endl;

    init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Dynamic6(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Dynamic6:"<< time/1000<<" ms"<<endl;       
       
	init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Dynamic4(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Dynamic4:"<< time/1000<<" ms"<<endl;
    
    init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Guided8(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Guide8:"<< time/1000<<" ms"<<endl;
        
    init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Guided6(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Guide6:"<< time/1000<<" ms"<<endl;
    
    init(test,n[i]);
    gettimeofday(&beg1,NULL);
    OpenMP_Guided4(test,n[i]);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"OpenMP_Guide4:"<< time/1000<<" ms"<<endl;

    }
    return 0;
}
