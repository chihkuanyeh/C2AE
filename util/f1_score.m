function [micro_f1, macro_f1] = f1_score(p, y)
	y = y > 0;
	p = p > 0;
	
	I = p .* y;
	U = p + y;
	
	micro_f1 = 2 * sum(sum(I)) / sum(sum(U));
	if isnan(micro_f1), micro_f1 = 1; end
	
	f = 2 * sum(I) ./ sum(U);
	f(isnan(f)) = [];
	macro_f1 = mean(f);
end
