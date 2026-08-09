/*
This is a simple genetic algorithm implementation where the
fitness of an individual depends only on the
objective function (it is the same as the value of the
objective function)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Change any of these parameters to match your needs */
#define POPSIZE 50     /* population size */
#define MAXGENS 100    /* no. of generations */
#define NVARS 3        /* no. of problem variables */
#define PXOVER 0.8     /* probability of crossover */
#define PMUTATION 0.15 /* probability of mutation */
#define TRUE 1
#define FALSE 0

int generation; /* current generation no. */
int cur_best;   /* best individual */
FILE *galog;    /* output file */

struct genotype /* genotype (GT), a member of the population */
{
    double gene[NVARS];  /* a string of variables */
    double upper[NVARS]; /* GT's variables upper bound */
    double lower[NVARS]; /* GT's variables lower bound */
    double fitness;      /* fitness of GT */
    double rfitness;     /* relative fitness */
    double cfitness;     /* cumulative fitness */
};

struct genotype population[POPSIZE + 1];    /* population */
struct genotype newpopulation[POPSIZE + 1]; /* new population */

/* Declaration of procedures used by this genetic algorithm */
void initialize(void);
double randval(double, double);
void evaluate(void);
void keep_the_best(void);
void elitist(void);
void select(void);
void crossover(void);
void mutate(void);
void report(void);

/*************************************************************/
/* Initialization function: Initializes the values of genes */
/* within the variables bounds. It also initializes (to zero)*/
/* all fitness values for each member of the population. It  */
/* reads upper and lower bounds of each variable from the    */
/* input file `gadata.txt'. It randomly generates values    */
/* between these bounds for each gene of each genotype in the*/
/* population. The format of the input file gadata.txt is   */
/* var1_lower_bound var1_upper_bound ...                    */
/* var2_lower_bound var2_upper_bound ...                     */
/*************************************************************/
void initialize(void)
{
    FILE *infile;
    int i, j;
    double lbound, ubound;

    if ((infile = fopen("gadata.txt", "r")) == NULL)
    {
        fprintf(galog, "\nCannot open input file!\n");
        exit(1);
    }

    /* initialize variables within the bounds */
    for (i = 0; i < NVARS; i++)
    {
        fscanf(infile, "%lf %lf", &lbound, &ubound);
        for (j = 0; j < POPSIZE; j++)
        {
            population[j].fitness = 0;
            population[j].rfitness = 0;
            population[j].cfitness = 0;
            population[j].lower[i] = lbound;
            population[j].upper[i] = ubound;
            population[j].gene[i] = randval(population[j].lower[i], population[j].upper[i]);
        }
    }
    fclose(infile);
}

/*************************************************************/
/* Random value generator: Generates a value within bounds   */
/*************************************************************/
double randval(double low, double high)
{
    double val;
    val = ((double)(rand() % 1000) / 1000.0) * (high - low) + low;
    return (val);
}

/*************************************************************/
/* Evaluation function: This takes a user defined function.  */
/* Each time this function is changed, the code has to be   */
/* re-compiled! The current function is: x[1]+2*x[2]+x[3]    */
/*************************************************************/
void evaluate(void)
{
    int mem;
    int i;
    double x[NVARS + 1];

    for (mem = 0; mem < POPSIZE; mem++)
    {
        for (i = 0; i < NVARS; i++)
            x[i + 1] = population[mem].gene[i];
        population[mem].fitness = (x[1]) + 2 * (x[2]) + (x[3]);
    }
}

/*************************************************************/
/* Keep the best function: This function keeps track of the  */
/* best member of the population. Note that the last entry in*/
/* the array population holds a copy of the best individual */
/*************************************************************/
void keep_the_best()
{
    int mem;
    int i;
    cur_best = 0; /* stores the index of the best individual */

    for (mem = 0; mem < POPSIZE; mem++)
    {
        if (population[mem].fitness > population[POPSIZE].fitness)
        {
            cur_best = mem;
            population[POPSIZE].fitness = population[mem].fitness;
        }
    }
    /* once the best member in the population is found, copy the genes */
    for (i = 0; i < NVARS; i++)
        population[POPSIZE].gene[i] = population[cur_best].gene[i];
}

/*************************************************************/
/* Elitist function: The best member of the previous         */
/* generation is stored as the last member of the array.    */
/* If the best member of the current generation is worse then*/
/* the best member of the previous generation then the      */
/* best one from the previous generation is put back into the*/
/* member of the current population                         */
/*************************************************************/
void elitist()
{
    int i;
    double best, worst;      /* best and worst fitness values */
    int best_mem, worst_mem; /* indexes of the best and worst member */

    best = population[0].fitness;
    worst = population[0].fitness;
    best_mem = 0;
    worst_mem = 0;

    for (i = 0; i < POPSIZE; i++)
    {
        if (population[i].fitness > best)
        {
            best = population[i].fitness;
            best_mem = i;
        }
        if (population[i].fitness < worst)
        {
            worst = population[i].fitness;
            worst_mem = i;
        }
    }

    /* if best individual from the new population is better than */
    /* the best individual from the previous population, then    */
    /* copy the best from the new population into the last place */
    if (best >= population[POPSIZE].fitness)
    {
        for (i = 0; i < NVARS; i++)
            population[POPSIZE].gene[i] = population[best_mem].gene[i];
        population[POPSIZE].fitness = population[best_mem].fitness;
    }
    else
    {
        for (i = 0; i < NVARS; i++)
            population[worst_mem].gene[i] = population[POPSIZE].gene[i];
        population[worst_mem].fitness = population[POPSIZE].fitness;
    }
}

/*************************************************************/
/* Selection function: Standard proportional selection for  */
/* maximization problems incorporating elitist model -- makes*/
/* sure that the best member survives                       */
/*************************************************************/
void select(void)
{
    int mem, i, j;
    double sum = 0;
    double p;

    /* find total fitness of the population */
    for (mem = 0; mem < POPSIZE; mem++)
    {
        sum += population[mem].fitness;
    }

    /* calculate relative fitness */
    for (mem = 0; mem < POPSIZE; mem++)
    {
        population[mem].rfitness = population[mem].fitness / sum;
    }

    /* calculate cumulative fitness */
    population[0].cfitness = population[0].rfitness;
    for (mem = 1; mem < POPSIZE; mem++)
    {
        population[mem].cfitness = population[mem - 1].cfitness + population[mem].rfitness;
    }

    /* finally select survivors using cumulative fitness. */
    for (i = 0; i < POPSIZE; i++)
    {
        // p = ((double)rand() % 1000) / 1000.0;
        p = (rand() % 1000) / 1000.0; //% 取模运算符不能直接用在 double 浮点数上
        if (p < population[0].cfitness)
            newpopulation[i] = population[0];
        else
        {
            for (j = 0; j < POPSIZE; j++)
            {
                if (p >= population[j].cfitness && p < population[j + 1].cfitness)
                    newpopulation[i] = population[j + 1];
            }
        }
    }

    /* once a new population is created, copy it back */
    for (i = 0; i < POPSIZE; i++)
    {
        population[i] = newpopulation[i];
    }
}

/*************************************************************/
/* Crossover function: select two parents that take part in */
/* the crossover, implements a single point crossover         */
/*************************************************************/
void crossover(void)
{
    int mem, one;
    int first = 0; /* count of the number of members chosen */
    double x;

    for (mem = 0; mem < POPSIZE; ++mem)
    {
        // x = ((double)rand() % 1000) / 1000.0;
        x = (rand() % 1000) / 1000.0; //% 取模运算符不能直接用在 double 浮点数上
        if (x < PXOVER)
        {
            ++first;
            if (first % 2 == 0)
            {
                /* perform crossover between mem and one */
                int point, i;
                point = rand() % NVARS;
                for (i = point; i < NVARS; i++)
                {
                    double temp = population[one].gene[i];
                    population[one].gene[i] = population[mem].gene[i];
                    population[mem].gene[i] = temp;
                }
            }
            else
            {
                one = mem;
            }
        }
    }
}

/*************************************************************/
/* Mutation function: uniform mutation. A variable selected  */
/* for mutation is replaced with a new value between lower and*/
/* upper bounds of that variable                            */
/*************************************************************/
void mutate(void)
{
    int i, j;
    double x;

    for (i = 0; i < POPSIZE; i++)
    {
        for (j = 0; j < NVARS; j++)
        {
            // x = ((double)rand() % 1000) / 1000.0;
            x = (rand() % 1000) / 1000.0;//% 取模运算符不能直接用在 double 浮点数上
            if (x < PMUTATION)
            {
                population[i].gene[j] = randval(population[i].lower[j], population[i].upper[j]);
            }
        }
    }
}

/*************************************************************/
/* Report function: prints out results                      */
/*************************************************************/
void report(void)
{
    int i;
    double best;
    double avg;
    double sum_sq;
    double stdev;

    /* compute statistics */
    sum_sq = 0.0;
    avg = 0.0;
    best = population[0].fitness;

    for (i = 0; i < POPSIZE; i++)
    {
        avg += population[i].fitness;
        sum_sq += population[i].fitness * population[i].fitness;
        if (population[i].fitness > best)
            best = population[i].fitness;
    }

    avg = avg / POPSIZE;
    stdev = sqrt(sum_sq / POPSIZE - avg * avg);

    fprintf(galog, "\n%3d %6.3f %6.3f %6.3f", generation, best, avg, stdev);
}

/*************************************************************/
/* Main function: Each generation involves selecting the best*/
/* members, performing crossover & mutation and then         */
/* evaluating the resulting population                       */
/*************************************************************/
int main(void)
{
    galog = fopen("galog.txt", "w");
    generation = 0;
    fprintf(galog, "\nGeneration Best Average Std Dev\n");
    fprintf(galog, "------------------------------------\n");
    initialize();
    evaluate();
    keep_the_best();
    report();

    while (generation < MAXGENS)
    {
        generation++;
        select();
        crossover();
        mutate();
        evaluate();
        elitist();
        report();
    }

    fclose(galog);
    return 0;
}