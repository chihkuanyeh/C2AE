function save_result(p, y, dataset, param1, param2, param3)
    [micro_f1, macro_f1] = f1_score(p, y);
    fprintf('micro_f1 = %f, macro_f1 = %f\n', micro_f1, macro_f1);
    score = micro_f1 + macro_f1;	

    fid = fopen([dataset, '_val_result.csv'], 'a');
    fprintf(fid, '%.1f,%.1f,%.1f,%f\n', param1, param2, param3, score);
    fclose(fid);
end
