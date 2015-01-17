#include "bp.h"
#include "memory.h"
#include "math.h"
#include "time.h"
#include "stdlib.h"
#include "stdio.h"

double sigmoid(double x)
{
    return 1 / (1 + exp(- x));
}

BpNet::BpNet(int nInput, int nHide, int nOutput)
{
    _nInput = nInput;
    _nHide = nHide;
    _nOutput = nOutput;
    
    srand((unsigned)time(NULL));
    
    _pplfWeight1 = new double *[_nInput];
    for (int i = 0; i < _nInput; i++)
    {
        _pplfWeight1[i] = new double [_nHide];
        for (int j = 0; j < _nHide; j++) _pplfWeight1[i][j] = rand() / (double)(RAND_MAX);
    }
    
    _pplfWeight2 = new double *[_nHide];
    for (int i = 0; i < _nHide; i++)
    {
        _pplfWeight2[i] = new double [_nOutput];
        for (int j = 0; j < _nOutput; j++) _pplfWeight2[i][j] = rand() / (double)(RAND_MAX);
    }
    
    _plfHideIn = new double[_nHide];
    _plfHideOut = new double[_nHide];
    _plfOutputIn = new double[_nOutput];
    _plfOutputOut = new double[_nOutput];
    f = sigmoid;    //使用sigmoid激活函数
}

BpNet::~BpNet()
{
    delete []_plfHideIn;
    delete []_plfHideOut;
    delete []_plfOutputIn;
    delete []_plfOutputOut;
    for (int i = 0; i < _nInput; i++) delete []_pplfWeight1[i];
    for (int i = 0; i < _nHide; i++) delete []_pplfWeight2[i];
    delete []_pplfWeight1;
    delete []_pplfWeight2;
}

void BpNet::Classify(double plfInput[], double plfOutput[])
{
    memset(_plfHideIn, 0, sizeof(double) * _nHide);
    memset(_plfHideOut, 0, sizeof(double) * _nHide);
    memset(_plfOutputIn, 0, sizeof(double) * _nOutput);
    memset(_plfOutputOut, 0, sizeof(double) * _nOutput);
    
    //输入层到隐含层的正向传播
    for (int i = 0; i < _nInput; i++)
        for (int j = 0; j < _nHide; j++)
            _plfHideIn[j] += plfInput[i] * _pplfWeight1[i][j];
    for (int j = 0; j < _nHide; j++) _plfHideOut[j] = (*f)(_plfHideIn[j]);
    
    //隐含层到输出层的正向传播
    for (int j = 0; j < _nHide; j++)
        for (int k = 0; k < _nOutput; k++)
            _plfOutputIn[k] += _plfHideOut[j] * _pplfWeight2[j][k];
    for (int k = 0; k < _nOutput; k++) _plfOutputOut[k] = (*f)(_plfOutputIn[k]);
    
    if (plfOutput != NULL)
        memcpy(plfOutput, _plfOutputOut, sizeof(double) * _nOutput);
}

bool BpNet::Train(int n, double **pplfInput, double **pplfDesire)
{
    const double a = 0.1;
    const double E = 0.01;
    double lfE = E + 1;
    
    //输入层-隐含层权系数增量
    double **pplfDeltaWeight1 = new double *[_nInput];
    for (int i = 0; i < _nInput; i++)
    {
        pplfDeltaWeight1[i] = new double [_nHide];
        memset(pplfDeltaWeight1[i], 0, sizeof(double) * _nHide);
    }
    
    //隐含层-输出层权系数增量
    double **pplfDeltaWeight2 = new double *[_nHide];
    for (int i = 0; i < _nHide; i++)
    {
        pplfDeltaWeight2[i] = new double[_nOutput];
        memset(pplfDeltaWeight2[i], 0, sizeof(double) * _nOutput);
    }
        
    int nCount = 0;
    while (lfE > E)
    {
        lfE = 0;
        //对每一个样本进行处理
        for (int i = 0; i < n; i++)
        {
            double *plfInput = pplfInput[i];        //样本输入
            double *plfDesire = pplfDesire[i];      //样本期望输出
        
            //计算样本实际输出plfOutput
            Classify(plfInput, NULL);
        
            //计算误差测度
            double lfEp = 0;
            for (int j = 0; j < _nOutput; j++)
                lfEp += (plfDesire[j] - _plfOutputOut[j]) * (plfDesire[j] - _plfOutputOut[j]);
            lfE += lfEp;
        
            //计算隐含层-输出层权系数增量
            double *plfChange2 = new double[_nOutput];
        
            for (int j = 0; j < _nOutput; j++)
                plfChange2[j] = _plfOutputOut[j] * (1 - _plfOutputOut[j]) * (plfDesire[j] - _plfOutputOut[j]);
            for (int j = 0; j < _nHide; j++)
                for (int k = 0; k < _nOutput; k++)
                    pplfDeltaWeight2[j][k] = a * _plfHideOut[j] * plfChange2[k];

            //计算输入层-隐含层权系数增量
            double *plfChange1 = new double[_nHide];
            memset(plfChange1, 0, sizeof(double) * _nHide);
            for (int j = 0; j < _nHide; j++)
            {
                for (int k = 0; k < _nOutput; k++)
                    plfChange1[j] += _pplfWeight2[j][k] * plfChange2[k];
                plfChange1[j] *= _plfHideOut[j] * (1 - _plfHideOut[j]);
            }
            for (int j = 0; j < _nInput; j++)
                for (int k = 0; k < _nHide; k++)
                    pplfDeltaWeight1[j][k] = a * plfInput[j] * plfChange1[k];
                
            delete []plfChange1;
            delete []plfChange2;
        
            //更新Bp网络权值
            for (int i = 0; i < _nInput; i++)
                for (int j = 0; j < _nHide; j++)
                    _pplfWeight1[i][j] += pplfDeltaWeight1[i][j];

            for (int i = 0; i < _nHide; i++)
                for (int j = 0; j < _nOutput; j++)
                    _pplfWeight2[i][j] += pplfDeltaWeight2[i][j];
        }
        nCount++;
        if (nCount % 10 == 0) printf("%d%8.5lf\n", nCount, lfE);
    }
            
    for (int i = 0; i < _nInput; i++) delete []pplfDeltaWeight1[i];
    for (int i = 0; i < _nHide; i++) delete []pplfDeltaWeight2[i];
    delete [] pplfDeltaWeight1;
    delete [] pplfDeltaWeight2;
    return true;
}
