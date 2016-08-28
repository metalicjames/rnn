#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <iterator>
#include <random>

#include "rnn.h"

int main()
{
    RNN rnn(1000);

    std::vector<arma::vec> x;
    std::vector<arma::vec> y;

    for(std::vector<RNN::trainingStruct>::iterator it = rnn.trainingData.begin(); it != rnn.trainingData.end(); it++)
    {
        arma::vec input((*it).x.size());
        arma::vec output((*it).y.size());
        for(unsigned int i = 0; i < (*it).x.size(); i++)
        {
            input(i) = (*it).x[i];
            output(i) = (*it).y[i];
        }
        x.push_back(input);
        y.push_back(output);
    }

    //std::cout << "Expected loss: " << std::log(1000) << std::endl;
    //std::cout << "Actual loss: " << rnn.calculateLoss(x, y);

    /*std::vector<arma::mat> output = rnn.bptt(x[0], y[0]);
    output[0].print();
    output[1].print();
    output[2].print();*/

    //rnn.gradient_check(x[0], y[0]);

    rnn.train_with_sgd(x, y);

    return 0;
}

std::string RNN::tokenFromId(unsigned int id)
{
    for(std::map<std::string, unsigned int>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++)
    {
        if(it->second == id)
        {
            return it->first;
        }
    }

    return "UNKNOWN_TOKEN";
}

unsigned int RNN::idFromToken(std::string token)
{
    std::map<std::string, unsigned int>::iterator it = vocabulary.find(token);
    if(it != vocabulary.end())
    {
        return it->second;
    }
    else
    {
        return vocabulary["UNKNOWN_TOKEN"];
    }
}

void RNN::loadVocabulary()
{
    std::ifstream ifs("tokens.txt");

    std::string word;

    std::map<std::string, unsigned int> dictionary;

    while(std::getline(ifs, word, ' '))
    {
        word.erase(std::remove(word.begin(), word.end(), '\n'), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        std::map<std::string, unsigned int>::iterator it = dictionary.find(word);
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
    std::sort(dicVec.begin(), dicVec.end(), [](std::pair<std::string, unsigned int> first, std::pair<std::string, unsigned int> second)
    {
        return first.second > second.second;
    });
    dicVec.resize(word_dim - 3);

    unsigned int i = 0;
    for(std::vector<std::pair<std::string, unsigned int>>::iterator it = dicVec.begin(); it != dicVec.end(); it++)
    {
        vocabulary.insert(std::pair<std::string, unsigned int> ((*it).first, i));
        i++;
    }

    vocabulary.insert(std::pair<std::string, unsigned int> ("SENTENCE_START", i));
    vocabulary.insert(std::pair<std::string, unsigned int> ("SENTENCE_END", i + 1));
    vocabulary.insert(std::pair<std::string, unsigned int> ("UNKNOWN_TOKEN", i + 2));

    std::vector<std::vector<unsigned int>> sentences;

    ifs.open("tokens.txt");

    std::string sentence;
    while(std::getline(ifs, sentence))
    {
        sentence.erase(std::remove(sentence.begin(), sentence.end(), '\n'), sentence.end());
        std::transform(sentence.begin(), sentence.end(), sentence.begin(), ::tolower);
        std::istringstream buf(sentence);
        std::istream_iterator<std::string> beg(buf), end;

        std::vector<std::string> tokens(beg, end);

        std::vector<unsigned int> ids;
        ids.push_back(vocabulary["SENTENCE_START"]);
        for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); it++)
        {
            ids.push_back(idFromToken(*it));
        }
        ids.push_back(vocabulary["SENTENCE_END"]);

        sentences.push_back(ids);
    }

    for(std::vector<std::vector<unsigned int>>::iterator it = sentences.begin(); it != sentences.end(); it++)
    {
        trainingStruct data;

        data.x = *it;
        data.y = *it;

        data.x.resize(data.x.size() - 1);
        data.y.erase(data.y.begin());

        trainingData.push_back(data);
    }
}

RNN::RNN(unsigned int nword_dim, unsigned int nhidden_dim, unsigned int nbptt_truncate)
{
    word_dim = nword_dim;
    hidden_dim = nhidden_dim;
    bptt_truncate = nbptt_truncate;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> Udist(-1/std::sqrt(word_dim), 1/std::sqrt(word_dim));
    std::uniform_real_distribution<double> Vdist(-1/std::sqrt(hidden_dim), 1/std::sqrt(hidden_dim));

    bool status = U.load("layers/U.dat");

    if(status)
    {
        V.load("layers/V.dat");
        W.load("layers/W.dat");
    }
    else
    {
        U.set_size(hidden_dim, word_dim);
        U.imbue( [&]()
        {
            return Udist(generator);
        });

        V.set_size(word_dim, hidden_dim);
        V.imbue( [&]()
        {
            return Vdist(generator);
        });

        W.set_size(hidden_dim, hidden_dim);
        W.imbue( [&]()
        {
            return Vdist(generator);
        });
    }

    loadVocabulary();
}

RNN::~RNN()
{
    W.save("layers/W.dat");
    U.save("layers/U.dat");
    V.save("layers/V.dat");
}

std::vector<arma::mat> RNN::forward_propagation(arma::vec x)
{
    unsigned int T = x.n_rows;

    arma::mat s(hidden_dim, T + 2, arma::fill::zeros);
    arma::mat o(word_dim, T, arma::fill::zeros);

    for(unsigned int t = 0; t < T; t++)
    {
        arma::mat result = U.col(x(t)) + (W * s.col(t));
        result.transform( [](double val)
        {
            return std::tanh(val);
        });
        s.col(t + 1) = result;

        //Calculate softmax
        result = arma::exp(V * s.col(t + 1));
        double sum = arma::accu(result);

        result.transform( [&](double val)
        {
            return val / sum;
        });
        o.col(t) = result;
    }

    std::vector<arma::mat> returning;
    returning.push_back(o);
    returning.push_back(s);

    return returning;
}

arma::urowvec RNN::predict(arma::vec x)
{
    arma::mat returning = forward_propagation(x)[0];
    return arma::index_max(returning, 0);
}

double RNN::calculateLoss(std::vector<arma::vec> x, std::vector<arma::vec> y)
{
    double L = 0;
    double N = 0;

    for(unsigned int i = 0; i < y.size(); i++)
    {
        N += y[i].n_rows;

        arma::mat o = arma::log(forward_propagation(x[i])[0]);

        for(unsigned int i2 = 0; i2 < y[i].n_rows; i2++)
        {
            L += -1 * o(y[i](i2), i2);
        }
    }

    return L / N;
}

std::vector<arma::mat> RNN::bptt(arma::vec x, arma::vec y)
{
    unsigned int T = y.n_rows;

    std::vector<arma::mat> output = forward_propagation(x);
    arma::mat o = output[0];
    arma::mat s = output[1];

    arma::mat dLdU(arma::size(U), arma::fill::zeros);
    arma::mat dLdV(arma::size(V), arma::fill::zeros);
    arma::mat dLdW(arma::size(W), arma::fill::zeros);

    arma::mat delta_o = o;
    for(unsigned int i = 0; i < y.n_rows; i++)
    {
        delta_o(y(i), i) -= 1;
    }

    for(int t = T - 1; t >= 0; t--)
    {
        dLdV += delta_o.col(t) * s.col(t + 1).t();
        arma::vec delta_t = (V.t() * delta_o.col(t)) % (1 - arma::square(s.col(t + 1)));
        int truncate = t - bptt_truncate;
        for(int bptt_step = t; bptt_step >= std::max(0, truncate); bptt_step--)
        {
            dLdW += delta_t * s.col(bptt_step).t();
            dLdU.col(x(bptt_step)) += delta_t;

            delta_t = (W.t() * delta_t) % (1 - arma::square(s.col(bptt_step)));
        }
    }

    std::vector<arma::mat> returning;
    returning.push_back(dLdU);
    returning.push_back(dLdV);
    returning.push_back(dLdW);

    return returning;
}

void RNN::gradient_check(arma::vec x, arma::vec y, double h, double error_threshold)
{
    std::vector<arma::mat> bptt_gradients = bptt(x, y);

    std::vector<arma::mat*> matrices;
    matrices.push_back(&U);
    matrices.push_back(&V);
    matrices.push_back(&W);

    for(std::vector<arma::mat*>::iterator it = matrices.begin(); it != matrices.end(); it++)
    {
        for(arma::mat::iterator elem = (*it)->begin(); elem != (*it)->end(); elem++)
        {
            double original_value = (*elem);

            (*elem) = original_value + h;
            double gradplus = calculateLoss(std::vector<arma::vec>(1, x), std::vector<arma::vec>(1, y));

            (*elem) = original_value - h;
            double gradminus = calculateLoss(std::vector<arma::vec>(1, x), std::vector<arma::vec>(1, y));

            double estimated_gradient = (gradplus - gradminus) / (2 * h);

            (*elem) = original_value;

            double backprop_gradient = bptt_gradients[std::distance(matrices.begin(), it)](std::distance((*it)->begin(), elem));

            double relative_error = std::abs(backprop_gradient - estimated_gradient) / (std::abs(backprop_gradient) + std::abs(estimated_gradient));

            if(relative_error > error_threshold)
            {
                std::cout << "Gradient Check ERROR" << std::endl;
                return;
            }
        }
    }

    std::cout << "Gradient check passed" << std::endl;
}

void RNN::sgd_step(arma::vec x, arma::vec y, double learning_rate)
{
    std::vector<arma::mat> bptt_result = bptt(x, y);

    U -= learning_rate * bptt_result[0];
    V -= learning_rate * bptt_result[1];
    W -= learning_rate * bptt_result[2];
}

void RNN::train_with_sgd(std::vector<arma::vec> x, std::vector<arma::vec> y, double learning_rate, unsigned int nepoch, unsigned int evaluate_loss_after)
{
    std::vector<std::array<double, 2>> losses;
    unsigned int num_examples_seen = 0;

    for(unsigned int epoch = 0; epoch < nepoch; epoch++)
    {
        if(epoch % evaluate_loss_after == 0)
        {
            double loss = calculateLoss(x, y);
            losses.push_back({static_cast<double>(num_examples_seen), loss});

            time_t tt = std::time(0);
            struct tm * ptm = std::localtime(&tt);
            char buf[20];
            std::strftime(buf, 20, "%Y-%m-%d %H:%M:%S", ptm);

            std::cout << buf << ": Loss after num_examples_seen = " << num_examples_seen << " epoch = " << epoch << ": " << loss << std::endl;
            if(losses.size() > 1 && losses[losses.size() - 1][1] > losses[losses.size() - 2][1])
            {
                learning_rate *= 0.5;
                std::cout << "Setting learning rate to " << learning_rate << std::endl;
            }
        }

        for(unsigned int i = 0; i < y.size(); i++)
        {
            sgd_step(x[i], y[i], learning_rate);
            num_examples_seen++;
        }
    }
}

