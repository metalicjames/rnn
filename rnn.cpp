#include <cmath>

#include "rnn.h"

RNN::RNN(unsigned int nword_dim, unsigned int nhidden_dim, unsigned int nbptt_truncate)
{
    word_dim = nword_dim;
    hidden_dim = nhidden_dim;
    bptt_truncate = nbptt_truncate;

    U.set_size(hidden_dim, word_dim);
    U.randn();

    V.set_size(word_dim, hidden_dim);
    V.randn();

    W.set_size(hidden_dim, hidden_dim);
    W.randn();
}

std::vector<arma::mat> RNN::forward_propagation(arma::vec x)
{
    unsigned int T = x.n_cols;

    arma::mat s(T + 2, hidden_dim, arma::fill::zeros);
    arma::mat o(T, word_dim);

    for(unsigned int t = 0; t < T; t++)
    {
        arma::mat result = U(x(t)) + (W * s(t));
        result.transform( [](double val) { return std::tanh(val); });
        s.col(t + 1) = result;

        //Calculate softmax
        result = V * s(t + 1);
        arma::mat temp = result;
        temp.transform( [](double val) { return std::exp(val); });
        double sum = arma::accu(temp);

        result.transform( [&](double val) { return std::exp(val) / sum; });
        o.col(t) = result;
    }

    std::vector<arma::mat> returning;
    returning.push_back(o);
    returning.push_back(s);

    return returning;
}
