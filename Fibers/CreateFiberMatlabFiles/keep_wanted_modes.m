% delete all the other modes file, and rename the wanted mode in the same
% order as on mode_list
% e.g: mode_list = [1, 4, 11] to mode 1 will be mode 1 and mode 4 will be
% mode 2 and mode 11 will be mode 3
function keep_wanted_modes(data,mode_list)

    files = dir(data.folder_name);
    cd(data.folder_name);
    for i=1:length(files)
        
        % file starts with "radius"
        if ~isempty(regexp(files(i).name, '^radius', 'once'))
            % get the mode number of the file
           mode_num = str2num((cell2mat(regexp(files(i).name, 'mode(\d+)', 'tokens', 'once'))));
           mode_strindex = regexp(files(i).name, 'mode') + 4; % +4 for after the word mode

           % check if we want this mode, if not, delet it
            if ~any(mode_list == mode_num)
                delete( files(i).name );
            
            % if we want, update the name
            else
                new_mode_number = find(mode_list == mode_num);
                new_name = files(i).name;
                new_name(mode_strindex:mode_strindex+2) = num2str(new_mode_number,'%03.f');

                if strcmp(files(i).name, new_name)
                    continue;
                end
               
                % change name
                movefile(files(i).name, new_name);

            end
            
        end
    end

cd('..');

end