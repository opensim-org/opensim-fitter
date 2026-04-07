import os
import numpy as np
import opensim as osim
import ezc3d
import scipy

class C3D:
    def __init__(self, filepath, columns_to_ignore=[], label_map={}):
        self.filepath = filepath
        self.c3d = ezc3d.c3d(filepath)

        # This is a Y-Z space-fixed rotation needed to convert data collected from Theia
        # to OpenSim's ground reference frame convention (X forward, Y up, Z right).
        osim_rotation = osim.Rotation()
        osim_rotation.setRotationFromTwoAnglesTwoAxes(1, # space-fixed
                -0.5*np.pi, osim.CoordinateAxis(1), # Y rotation
                -0.5*np.pi, osim.CoordinateAxis(2)) # Z rotation
        self.osim_rotation = osim_rotation

        # This is an additional body-fixed rotation that effectively swaps the axes of
        # the rotations collected from Theia to match OpenSim's ground reference frame
        # convention (X forward, Y up, Z right), which is the convention used by the
        # matching Frame elements in the generic model.
        frame_rotation = osim.Rotation()
        frame_rotation.setRotationToBodyFixedXY(osim.Vec2(0.5*np.pi))
        self.frame_rotation = frame_rotation

        # Columns that should be ignored when processing the data.
        self.columns_to_ignore = columns_to_ignore

        # Map of original labels to new labels.
        self.label_map = label_map

    def get_data(self, parameter):
        return self.c3d.data[parameter]

    def get_data_labels(self, parameter):
        raw_labels = self.c3d.parameters[parameter]['LABELS']['value']
        labels = [label.replace('_4X4', '') for label in raw_labels]
        return labels

    def get_data_rate(self, parameter):
        return self.c3d.parameters[parameter]['RATE']['value'][0]

    def get_time_vector(self, rate, num_frames):
        return np.array([i/rate for i in range(num_frames)])

    def remove_ignored_columns(self, table):
        for col in self.columns_to_ignore:
            table.removeColumn(col)
        return table

    def update_column_labels(self, table):
        if not self.label_map:
            return table

        labels = list(table.getColumnLabels())
        for ilabel in range(len(labels)):
            labels[ilabel] = self.label_map.get(labels[ilabel])
        table.setColumnLabels(labels)
        return table

    def get_positions_table(self):
        data = self.get_data('rotations')
        num_frames = data.shape[3]
        labels = self.get_data_labels('ROTATION')
        rate = self.get_data_rate('ROTATION')
        times = self.get_time_vector(rate, num_frames)

        table = osim.TimeSeriesTableVec3()
        for iframe in range(num_frames):
            row = osim.RowVectorVec3(len(labels), osim.Vec3(0))
            for ilabel, label in enumerate(labels):
                position = data[:, 3, ilabel, iframe] / 1000.0  # mm to m
                row[ilabel] = osim.Vec3(position[0], position[1], position[2])
                row[ilabel] = self.osim_rotation.multiply(row[ilabel])

            table.appendRow(times[iframe], row)

        table.setColumnLabels(labels)
        table = self.remove_ignored_columns(table)
        table = self.update_column_labels(table)
        table.addTableMetaDataString("Units", "m")
        table.addTableMetaDataString("DataRate", str(rate))

        return table

    def get_quaternions_table(self):
        data = self.get_data('rotations')
        num_frames = data.shape[3]
        labels = self.get_data_labels('ROTATION')
        rate = self.get_data_rate('ROTATION')
        times = self.get_time_vector(rate, num_frames)

        table = osim.TimeSeriesTableQuaternion()
        for iframe in range(num_frames):
            row = osim.RowVectorQuaternion(len(labels), osim.Quaternion())
            for ilabel, label in enumerate(labels):
                rot = data[:3, :3, ilabel, iframe]
                data_rotation = osim.Rotation()
                data_rotation.set(0,0, rot[0,0])
                data_rotation.set(1,0, rot[1,0])
                data_rotation.set(2,0, rot[2,0])
                data_rotation.set(0,1, rot[0,1])
                data_rotation.set(1,1, rot[1,1])
                data_rotation.set(2,1, rot[2,1])
                data_rotation.set(0,2, rot[0,2])
                data_rotation.set(1,2, rot[1,2])
                data_rotation.set(2,2, rot[2,2])
                rotation = self.osim_rotation.multiply(data_rotation)
                rotation = rotation.multiply(self.frame_rotation)

                # Store as a quaternion.
                new_quat = rotation.convertRotationToQuaternion()
                upd_quat = row.updElt(0, ilabel)
                upd_quat.set(0, new_quat.get(0))
                upd_quat.set(1, new_quat.get(1))
                upd_quat.set(2, new_quat.get(2))
                upd_quat.set(3, new_quat.get(3))

            table.appendRow(times[iframe], row)

        table.setColumnLabels(labels)
        table = self.remove_ignored_columns(table)
        table = self.update_column_labels(table)
        table.addTableMetaDataString("DataRate", str(rate))

        return table


class MultivariateNormal:
    @classmethod
    def from_data(cls, variables, data):
        """
        Create a MultivariateNormal from data.

        Parameters
        ----------
        variables : list of str
            The names of the variables in the distribution.
        data : ndarray (m, n)
            Data the distribution is fitted to.

        Returns
        -------
        MultivariateNormal
            The fitted distribution.
        """
        mean, cov = scipy.stats.multivariate_normal.fit(data)
        return cls(variables, mean, cov)

    def __init__(self, variables, mean, cov):
        """
        Create a MultivariateNormal directly from the mean and covariance.

        Parameters
        ----------
        variables : list of str
            The names of the variables in the distribution.
        mean : ndarray (n,)
            The mean of the distribution.
        cov : ndarray (n, n)
            The covariance matrix of the distribution.

        Returns
        -------
        MultivariateNormal
            The created distribution.
        """
        assert(len(variables) == mean.shape[0] == cov.shape[0] == cov.shape[1])
        self.variables = variables
        self.mean = mean
        self.cov = cov
        self.multivariate_normal = scipy.stats.multivariate_normal(
            mean=self.mean, cov=self.cov)

    def _convert_values_from_dict(self, values):
        """
        Convert a dictionary of variable values to an ndarray.

        Parameters
        ----------
        values : dict
            A dictionary mapping variable names to their values.

        Returns
        -------
        ndarray (n,)
            The values in the order of self.variables.
        """
        x = np.zeros(len(self.variables))
        for var in values.keys():
            if var not in self.variables:
                raise ValueError(f'Variable {var} not in distribution.')
            x[self.variables.index(var)] = values[var]
        return x

    def get_variables(self):
        """
        Get the names of the variables in the distribution.

        Returns
        -------
        list of str
            The names of the variables.
        """
        return self.variables

    def get_mean(self):
        """
        Get the mean of the distribution.

        Returns
        -------
        ndarray (n,)
            The mean of the distribution.
        """
        return self.mean

    def get_covariance(self):
        """
        Get the covariance matrix of the distribution.

        Returns
        -------
        ndarray (n, n)
            The covariance matrix of the distribution.
        """
        return self.cov

    def get_dimension(self):
        """
        Get the dimensionality of the distribution.

        Returns
        -------
        int
            The number of variables in the distribution.
        """
        return len(self.variables)

    def get_random_sample(self):
        """
        Get a random sample from the distribution.

        Returns
        -------
        ndarray (n,)
            A random sample from the distribution.
        """
        return self.multivariate_normal.rvs()

    def get_pdf(self, values):
        """
        Get the probability density function (PDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the PDF.

        Returns
        -------
        float
            The PDF value at the given point.
        """
        return self.multivariate_normal.pdf(self._convert_values_from_dict(values))

    def get_logpdf(self, values):
        """
        Get the log of the probability density function (PDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the log PDF.

        Returns
        -------
        float
            The log PDF value at the given point.
        """
        return self.multivariate_normal.logpdf(self._convert_values_from_dict(values))

    def get_cdf(self, values):
        """
        Get the cumulative density function (CDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the CDF.

        Returns
        -------
        float
            The CDF value at the given point.
        """
        return self.multivariate_normal.cdf(self._convert_values_from_dict(values))

    def get_logcdf(self, values):
        """
        Get the log of the cumulative density function (CDF) value at a given point.

        Parameters
        ----------
        values : ndarray (n,)
            The point at which to evaluate the log CDF.

        Returns
        -------
        float
            The log CDF value at the given point.
        """
        return self.multivariate_normal.logcdf(self._convert_values_from_dict(values))

    def condition(self, values):
        """
        Condition the multivariate normal distribution on a subset of variables
        being equal to the specified 'values'. Returns a new MultivariateNormal instance
        for the conditioned distribution.

        Parameters
        ----------
        values : dict
            A dictionary mapping variable names to their observed values.

        Returns
        -------
        MultivariateNormal
            The conditioned distribution.
        """
        idx_b = [self.variables.index(var) for var in values.keys()]
        idx_a = [i for i in range(len(self.variables)) if i not in idx_b]

        mean_a = self.mean[idx_a]
        mean_b = self.mean[idx_b]

        cov_aa = self.cov[np.ix_(idx_a, idx_a)]
        cov_bb = self.cov[np.ix_(idx_b, idx_b)]
        cov_ab = self.cov[np.ix_(idx_a, idx_b)]
        cov_ba = self.cov[np.ix_(idx_b, idx_a)]

        cov_bb_inv = np.linalg.inv(cov_bb)

        values_array = np.array(list(values.values()))
        mean_cond = mean_a + cov_ab @ cov_bb_inv @ (values_array - mean_b)
        cov_cond = cov_aa - cov_ab @ cov_bb_inv @ cov_ba

        variables_a = [self.variables[i] for i in idx_a]
        return MultivariateNormal(variables_a, mean_cond, cov_cond)

    def get_variable_percentile(self, variable_name, value):
        """
        Calculate the percentile rank of a specific value for an individual variable.

        Parameters
        ----------
        variable_name : str
            The name of the variable.
        value : float
            The value to find the percentile for.

        Returns
        -------
        float
            The percentile rank (0-100) of the given value.

        Raises
        ------
        ValueError
            If the variable is not in the distribution.
        """
        if variable_name not in self.variables:
            raise ValueError(f"Variable '{variable_name}' not found in the distribution")

        # Get index of the variable
        var_idx = self.variables.index(variable_name)

        # Get mean and standard deviation for this variable
        var_mean = self.mean[var_idx]
        var_std = np.sqrt(self.cov[var_idx, var_idx])

        # Create a normal distribution for this variable
        var_dist = scipy.stats.norm(var_mean, var_std)

        # Calculate the percentile (CDF * 100)
        percentile = var_dist.cdf(value) * 100.0

        return percentile

    def plot_variable_distribution(self, data_df, variable_name, bins=30):
        """
        Plot the fitted distribution versus actual data for a specific variable.

        Parameters
        ----------
        data_df : pandas.DataFrame
            The original data
        variable_name : str
            Name of the variable to plot
        bins : int, optional
            Number of histogram bins (default: 30)
        """
        import matplotlib.pyplot as plt

        # Check if variable exists in the distribution
        if variable_name not in self.variables:
            raise ValueError(f"Variable '{variable_name}' not found in the distribution")

        # Get index of the variable
        var_idx = self.variables.index(variable_name)

        # Get mean and standard deviation for this variable
        var_mean = self.mean[var_idx]
        var_std = np.sqrt(self.cov[var_idx, var_idx])

        # Create a normal distribution for this variable
        var_dist = scipy.stats.norm(var_mean, var_std)

        # Plot histogram of actual data
        plt.figure(figsize=(10, 6))
        plt.hist(data_df[variable_name], bins=bins, density=True, alpha=0.6,
                label='Actual data', color='skyblue')

        # Plot the fitted distribution
        x = np.linspace(var_mean - 4*var_std, var_mean + 4*var_std, 1000)
        plt.plot(x, var_dist.pdf(x), 'r-', lw=2, label='Fitted normal distribution')

        plt.title(f'Distribution of {variable_name}')
        plt.xlabel(variable_name)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def get_coordinate_indexes(model, skip_dependent_coordinates=True):
    """Get a mapping of coordinate paths to their indexes in the state vector.
    """
    state = model.getWorkingState()
    state_paths = osim.createStateVariableNamesInSystemOrder(model)
    coordinates_map = {}
    for i, state_path in enumerate(state_paths):
        if 'value' in state_path:
            coord_path = state_path.replace('/value', '')
            coordinate = osim.Coordinate.safeDownCast(model.getComponent(coord_path))
            if skip_dependent_coordinates:
                if not coordinate.isDependent(state):
                    coordinates_map[coord_path] = i
            else:
                coordinates_map[coord_path] = i

    return coordinates_map


def get_ipopt_options(convergence_tolerance=1e-4):
    """Get a dictionary of common IPOPT options for use with CasADi's nlpsolver.
    """
    ipopt_options = {}
    ipopt_options['hessian_approximation'] = 'limited-memory'
    ipopt_options['tol'] = convergence_tolerance
    ipopt_options['dual_inf_tol'] = convergence_tolerance
    ipopt_options['compl_inf_tol'] = convergence_tolerance
    ipopt_options['acceptable_tol'] = convergence_tolerance
    ipopt_options['acceptable_dual_inf_tol'] = convergence_tolerance
    ipopt_options['acceptable_compl_inf_tol'] = convergence_tolerance
    # ipopt_options['constr_viol_tol'] = constraint_tolerance
    # ipopt_options['acceptable_constr_viol_tol'] = constraint_tolerance
    ipopt_options['print_level'] = 0

    return ipopt_options
