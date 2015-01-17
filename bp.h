#ifndef BP_H
#define BP_H

class BpNet
{
private:
    int _nInput;            //输入层节点个数
    int _nHide;             //隐含层节点个数
    int _nOutput;           //输出层节点个数
    
    double **_pplfWeight1;  //输入层-隐含层权系数
    double **_pplfWeight2;  //隐含层-输出层权系数
    
    double *_plfHideIn, *_plfHideOut;       //隐含层的网络输入和输出
    double *_plfOutputIn, *_plfOutputOut;   //输出层的网络输入和输出
    
private:
    double (*f)(double);    //激活函数
    
public:
    /**
    构造函数，创建一个未经训练的Bp网络
    param nInput 输入层节点个数
    param nHide 隐含层节点个数
    param nOutput 输出层节点个数
    */
    BpNet(int nInput, int nHide, int nOutput);
    
    /**
    析构函数，销毁已有的Bp网络
    */
    virtual ~BpNet();
    
    /**
    通过多组正例，对Bp网络进行训练
    param pplfInput 正例的输入
    param pplfDesire 正例的理想输出
    返回值 训练是否收敛
    */
    bool Train(int n, double **pplfInput, double **pplfDesire);
    
    /**
    使用当前的Bp网络对指定模式进行分类,分类的结果存储到plfOutput中
    param plfInput 待分类的模式
    param plfOutput 分类的输出
    */
    void Classify(double plfInput[], double plfOutput[]);
};

#endif