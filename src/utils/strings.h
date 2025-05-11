//
// Created by vikimaster2 on 2/18/24.
//

#ifndef STRINGS_H
#define STRINGS_H
#include <algorithm>
#include <ranges>
#include <string>

inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::ranges::find_if(s, [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

#endif //STRINGS_H
