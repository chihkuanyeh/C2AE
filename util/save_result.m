function save_result(p, y, dataset, param1, param2, param3)
	[micro_f1, macro_f1] = f1_score(p, y);
	fprintf('micro_f1 = %f, macro_f1 = %f\n', micro_f1, macro_f1);
	
	fid = fopen([dataset, '_result.csv'], 'a');
    fprintf(fid, '%.1f,%.1f,%.1f,%f,%f\n', param1, param2, param3, micro_f1, macro_f1);
    fclose(fid);
end
