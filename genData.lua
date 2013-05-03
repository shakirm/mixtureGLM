-- Generate synthetic data for testing mixtures of GLM experts
-- Shakir, April 2013
require 'csv'

function genData(type,nObs,plot)

	if type == 'wave' then	
	-- Use the S-shaped data set for 3 expert test
		local y = torch.rand(nObs,1);
		local epsilon = torch.randn(nObs,1)*0.05;
		local x = torch.Tensor(nObs,1):zero();

		-- q = y + eta + 0.3*sin(2*piw*y)
		local q = torch.mul(torch.sin(torch.mul(y,2*math.pi)),0.3)
		x = y + q + epsilon;	
		
		if (1==plot) then
			gnuplot.figure(1)
			gnuplot.plot({torch.squeeze(x),torch.squeeze(y)});
			gnuplot.title('Input Data');
		end;
		
		return x, y;

	elseif type == 'teepee' then
		-- Use the upside down-U shape.
		local x = torch.Tensor(nObs,1):zero()
		local y = torch.Tensor(nObs,1):zero();
		
		for i = 1,nObs do
			local tmp = torch.rand(1);
			x[i][1] = tmp;
			
			if (torch.squeeze(tmp)<0.5) then
				y[i][1] = 2*torch.squeeze(tmp) + 0.1*torch.squeeze(torch.randn(1));
			else
				y[i][1] = 2 - 2*torch.squeeze(tmp) + 0.1*torch.squeeze(torch.randn(1));
			end;	
		end;
		
		if (1==plot) then
			gnuplot.figure(1)
			gnuplot.plot({torch.squeeze(x),torch.squeeze(y)});
			gnuplot.title('Input Data');
		end;
		
		return x, y;

	else
		print('Error: No such data set');
	end;
	
	return x, y;
end;