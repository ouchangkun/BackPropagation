#include "stdio.h"
#include "bp.h"

int main()
{
    int nInput = 2;
    int nHide = 2;
    int nOutput = 1;

    BpNet bpNet(nInput, nHide, nOutput);

    int n = 4;
    double **ppInput, **ppOutput;

    ppInput = new double *[n];
    ppOutput = new double *[n];

    for (int i = 0; i < n; i++)
    {
        ppInput[i] = new double[2];
        ppOutput[i] = new double[1];
    }
    /*ppInput[0][0] = 0.8;  ppInput[0][1] = 0.5; ppInput[0][2] = 0;
    ppInput[1][0] = 0.9;  ppInput[1][1] = 0.7; ppInput[1][2] = 0.3;
    ppInput[2][0] = 1;  ppInput[2][1] = 0.8; ppInput[2][2] = 0.5;
    ppInput[3][0] = 0;  ppInput[3][1] = 0.2; ppInput[3][2] = 0.3;
    ppInput[4][0] = 0.2;  ppInput[4][1] = 0.1; ppInput[4][2] = 1.3;
    ppInput[5][0] = 0.2;  ppInput[5][1] = 0.7; ppInput[5][2] = 0.8;
    
    ppOutput[0][0] = 0; ppOutput[0][1] = 1;
    ppOutput[1][0] = 0; ppOutput[1][1] = 1;
    ppOutput[2][0] = 0; ppOutput[2][1] = 1;
    ppOutput[3][0] = 1; ppOutput[3][1] = 0;
    ppOutput[4][0] = 1; ppOutput[4][1] = 0;
    ppOutput[5][0] = 1; ppOutput[5][1] = 0;
    */
    ppInput[0][0] = 0;  ppInput[0][1] = 0;
    ppInput[1][0] = 0;  ppInput[1][1] = 1;
    ppInput[2][0] = 1;  ppInput[2][1] = 0;
    ppInput[3][0] = 1;  ppInput[3][1] = 1;


    ppOutput[0][0] = 0;
    ppOutput[1][0] = 1;
    ppOutput[2][0] = 1;
    ppOutput[3][0] = 0;

    bpNet.Train(n, ppInput, ppOutput);

    double pTest[1];
    double pRs[1];

    while (1)
    {
        printf("test:\n");
        scanf("%lf%lf", &pTest[0], &pTest[1]);
        bpNet.Classify(pTest, pRs);
        printf("%lf\n", pRs[0]);
    }
    return 0;
}
