create or replace function mycount1()
returns integer as $$

declare
 mysql text;
 counts integer;

begin
-- CREATE TEMP TABLE log (ts timestamp DEFAULT clock_timestamp(), msg text);

mysql:='select count(*) from people';


COPY (select now(), 'execute：mysql' ) TO PROGRAM 'more >> /home/wang/work/log/fun.log';
-- insert into log(msg) values ('execute：' || mysql);
execute mysql into counts;
-- insert into log(msg) values ('execute over return count: ' || counts);
COPY (select now(), 'execute over return count: ') TO PROGRAM 'more >> /home/wang/work/log/fun.log';
-- COPY log TO PROGRAM 'more >> /home/wang/work/log/fun.log';
return counts;
end;

$$ language plpgsql;