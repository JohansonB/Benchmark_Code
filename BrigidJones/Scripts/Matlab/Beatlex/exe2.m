for c = 1:10
   [starts, ends, p_idx, p_starts, p_ends, models] = summarize_seq(X, 40+10*c, 200, 180,0)
end
disp("pepe")