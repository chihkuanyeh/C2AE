function save_more(p, y, dataset, param1, param2, param3)
	[C_P, C_R, C_F1, O_P, O_R, O_F1] = scores(p, y);
	fprintf('C_P = %f, C_R = %f, C_F1 = %f, O_P = %f, O_R = %f, O_F1 = %f\n', C_P, C_R, C_F1, O_P, O_R, O_F1);
	
	fid = fopen([dataset, '_result.csv'], 'a');
    fprintf(fid, '%.1f,%.1f,%.1f,%f,%f,%f,%f,%f,%f\n', param1, param2, param3, C_P, C_R, C_F1, O_P, O_R, O_F1);
    fclose(fid);
end
