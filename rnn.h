#ifndef RNN_H_INCLUDED
#define RNN_H_INCLUDED

#include <vector>

#include <armadillo>

class RNN
{
    public:
        RNN(unsigned int nword_dim, unsigned int nhidden_dim = 100, unsigned int nbptt_truncate = 4);
        arma::rowvec predict(arma::vec x);

    private:
        unsigned int word_dim;
        unsigned int hidden_dim;
        unsigned int bptt_truncate;
        arma::mat U;
        arma::mat V;
        arma::mat W;
        std::vector<arma::mat> forward_propagation(arma::vec x);
};

#endif // RNN_H_INCLUDED
