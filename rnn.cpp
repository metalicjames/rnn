#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <iterator>

#include "rnn.h"

int main()
{
    RNN rnn(1000);

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

    U.set_size(hidden_dim, word_dim);
    U.randn();

    V.set_size(word_dim, hidden_dim);
    V.randn();

    W.set_size(hidden_dim, hidden_dim);
    W.randn();

    loadVocabulary();
}

std::vector<arma::mat> RNN::forward_propagation(arma::vec x)
{
    unsigned int T = x.n_cols;

    arma::mat s(T + 2, hidden_dim, arma::fill::zeros);
    arma::mat o(T, word_dim, arma::fill::zeros);

    for(unsigned int t = 0; t < T; t++)
    {
        arma::mat result = U(x(t)) + (W * s(t));
        result.transform( [](double val)
        {
            return std::tanh(val);
        });
        s.col(t + 1) = result;

        //Calculate softmax
        result = V * s(t + 1);
        arma::mat temp = result;
        temp.transform( [](double val)
        {
            return std::exp(val);
        });
        double sum = arma::accu(temp);

        result.transform( [&](double val)
        {
            return std::exp(val) / sum;
        });
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
