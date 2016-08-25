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
        std::map<std::string, unsigned int> vocabulary;
        void loadVocabulary();
        std::string tokenFromId(unsigned int id);
        unsigned int idFromToken(std::string token);
        struct trainingStruct
        {
            std::vector<unsigned int> x;
            std::vector<unsigned int> y;
        };
        std::vector<trainingStruct> trainingData;
};

#endif // RNN_H_INCLUDED
