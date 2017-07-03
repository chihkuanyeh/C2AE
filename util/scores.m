function [C_P, C_R, C_F1, O_P, O_R, O_F1] = scores(p, y)
	y = y > 0;
	p = p > 0;
	
	tp = p .* y;
	fp = p .* (~y);
	fn = (~p) .* y;
	s = p + y;
	
	O_P = sum(sum(tp)) / sum(sum(tp + fp));	
	O_R = sum(sum(tp)) / sum(sum(tp + fn));
	O_F1 = 2 * sum(sum(tp)) / sum(sum(s));
	if isnan(O_F1), O_F1 = 1; end
	
	p = sum(tp) ./ sum(tp + fp);
	p(isnan(p)) = [];
	C_P = mean(p);
	r = sum(tp) ./ sum(tp + fn);
	r(isnan(r)) = [];
	C_R = mean(r);
	f = 2 * sum(tp) ./ sum(s);
	f(isnan(f)) = [];
	C_F1 = mean(f);
end
