    def _legend_format_fit_output(self) -> str:
        # pylint: disable=no-member
        """String formatting for legend fit output.

        Returns
        -------
        text : str
            The formatted string.
        """
        return f"FWHM@{self.energies[0]:.1f} keV: {self.fwhm():.1f} %"

    def _plot(self, axes: matplotlib.axes.Axes = None, fit_output: bool = False,
             **kwargs) -> matplotlib.axes.Axes:
        """Plot the model.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on (default: current axes).

        kwargs : dict, optional
            Additional keyword arguments passed to `plt.plot()`.
        """
        kwargs.setdefault("label", self.label)
        if fit_output:
            kwargs["label"] = f"{kwargs['label']}\n{self._legend_format_fit_output()}"
        return super(AbstractFitModel, self).plot(axes, **kwargs)
