#ifndef RNN_H_INCLUDED
#define RNN_H_INCLUDED

#include <vector>

#include <armadillo>

class RNN
{
    public:
        RNN(unsigned int nword_dim, unsigned int nhidden_dim = 100, unsigned int nbptt_truncate = 1000);
        ~RNN();
        arma::urowvec predict(arma::vec x);
        struct trainingStruct
        {
            std::vector<unsigned int> x;
            std::vector<unsigned int> y;
        };
        std::vector<trainingStruct> trainingData;
        std::vector<arma::mat> forward_propagation(arma::vec x);
        std::string tokenFromId(unsigned int id);
        unsigned int idFromToken(std::string token);
        double calculateLoss(std::vector<arma::vec> x, std::vector<arma::vec> y);
        std::vector<arma::mat> bptt(arma::vec x, arma::vec y);
        void gradient_check(arma::vec x, arma::vec y, double h = 0.001, double error_threshold = 0.01);
        void train_with_sgd(std::vector<arma::vec> x, std::vector<arma::vec> y, double learning_rate = 0.005, unsigned int nepoch = 100, unsigned int evaluate_loss_after = 5);
        std::vector<unsigned int> generate_sentence();

    private:
        unsigned int word_dim;
        unsigned int hidden_dim;
        unsigned int bptt_truncate;
        arma::mat U;
        arma::mat V;
        arma::mat W;
        std::map<std::string, unsigned int> vocabulary;
        void loadVocabulary();
        void sgd_step(arma::vec x, arma::vec y, double learning_rate);
};

#endif // RNN_H_INCLUDED
