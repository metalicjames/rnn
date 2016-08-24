#include <cmath>
#include <iostream>
#include <map>

#include "rnn.h"

int main()
{
    std::ifstream ifs("tokens.txt");

    std::string word;

    std::map<std::string, unsigned int> dictionary;
    std::map<std::string, unsigned int>::iterator it;

    while(std::getline(ifs, word, ' '))
    {
        word.erase(std::remove(word.begin(), word.end(), '\n'), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        it = dictionary.find(word);
        if(it != dictionary.end())
        {
            it->second++;
        }
        else
        {
            dictionary.insert(std::pair<std::string, unsigned int> (word, 1));
        }
    }

    ifs.close();

    std::vector<std::pair<std::string, unsigned int>> dicVec(dictionary.begin(), dictionary.end());
    std::sort(dicVec.begin(), dicVec.end(), [](std::pair<std::string, unsigned int> first, std::pair<std::string, unsigned int> second) { return first.second > second.second; });

    for(std::vector<std::pair<std::string, unsigned int>>::iterator it = dicVec.begin(); it != dicVec.end(); it++)
    {
        std::cout << (*it).first << ": " << (*it).second << std::endl;
    }

    return 0;
}

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

arma::rowvec RNN::predict(arma::vec x)
{
    arma::mat returning = forward_propagation(x)[0];
    return arma::max(returning, 0);
}
