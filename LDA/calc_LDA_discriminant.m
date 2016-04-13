function disc = calc_LDA_discriminant(x, means, C_inv, prob, class)
    disc = means(class, :)*C_inv*x' - .5*means(class, :)*C_inv*means(class, :)' + log(prob(class));
end