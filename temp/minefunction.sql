create or replace function mycount()
returns integer as $$

declare
 mysql text;
 counts integer;

begin
CREATE TEMP TABLE log (ts timestamp DEFAULT clock_timestamp(), msg text);

mysql:='select count(*) from people';

insert into log(msg) values ('execute：' || mysql);
execute mysql into counts;
insert into log(msg) values ('execute over return count: ' || counts);

COPY log TO PROGRAM 'more >> /home/wang/work/log/fun.log';
return counts;
end;

$$ language plpgsql;