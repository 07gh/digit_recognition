function odds = calc_odds(x, beta)
    temp = exp(x * beta);
    odds = temp / (1+temp);
end