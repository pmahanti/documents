"""
Impact Crater Analysis Report Generator

Generates comprehensive PDF reports for impact crater analysis including:
- Impact parameter calculations with uncertainties
- Theoretical explanations and equations
- Excavation depth and ejecta volume estimates
- Detailed methodology and assumptions

Uses the impact_scaling module for core calculations.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

# Import our impact scaling module
from impact_scaling import ImpactScaling, MATERIALS, IMPACTORS


class ImpactReportGenerator:
    """
    Generate comprehensive PDF reports for impact crater analysis.
    """

    def __init__(self):
        """Initialize the report generator."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        # Helper function to add or update style
        def add_style(name, **kwargs):
            if name in self.styles:
                # Update existing style
                style = self.styles[name]
                for key, value in kwargs.items():
                    setattr(style, key, value)
            else:
                # Add new style
                self.styles.add(ParagraphStyle(name=name, **kwargs))

        # Title style
        add_style('CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # Section heading
        add_style('SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        # Subsection heading
        add_style('SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )

        # Body text
        add_style('BodyText',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=16,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )

        # Equation style
        add_style('Equation',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Courier',
            leftIndent=30,
            spaceAfter=6,
            spaceBefore=6
        )

        # Caption style
        add_style('Caption',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=TA_CENTER
        )

    def compute_uncertainties(self, D: float, d: float, U: float,
                             D_err: float, d_err: float, U_err: float,
                             target_material: str, impactor_density: float,
                             crater_type: str = 'simple',
                             n_samples: int = 1000) -> Dict:
        """
        Compute uncertainties using Monte Carlo sampling.

        Parameters:
        -----------
        D, d, U : float
            Crater diameter, depth, velocity (nominal values)
        D_err, d_err, U_err : float
            Uncertainties in diameter, depth, velocity
        target_material : str
            Target material name
        impactor_density : float
            Impactor density (kg/m³)
        crater_type : str
            'simple' or 'complex'
        n_samples : int
            Number of Monte Carlo samples

        Returns:
        --------
        uncertainties : dict
            Dictionary with mean, std, and percentiles for each parameter
        """
        target = MATERIALS[target_material]
        calc = ImpactScaling(target, impactor_density)

        # Storage for samples
        samples = {
            'impactor_diameter': [],
            'impactor_mass': [],
            'impact_energy': [],
            'impact_velocity': []
        }

        # Monte Carlo sampling
        for _ in range(n_samples):
            # Sample from normal distributions
            D_sample = np.random.normal(D, D_err)
            d_sample = np.random.normal(d, d_err)
            U_sample = np.random.normal(U, U_err)

            # Ensure positive values
            D_sample = max(D_sample, 0.1)
            d_sample = max(d_sample, 0.01)
            U_sample = max(U_sample, 100)

            try:
                results = calc.compute_impactor_params(
                    D_sample, d_sample, U_sample, crater_type=crater_type
                )

                samples['impactor_diameter'].append(results['impactor_diameter'])
                samples['impactor_mass'].append(results['impactor_mass'])
                samples['impact_energy'].append(results['impact_energy'])
                samples['impact_velocity'].append(U_sample)
            except:
                continue

        # Compute statistics
        uncertainties = {}
        for key, values in samples.items():
            if len(values) > 0:
                arr = np.array(values)
                uncertainties[key] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'median': np.median(arr),
                    'p16': np.percentile(arr, 16),
                    'p84': np.percentile(arr, 84),
                    'p5': np.percentile(arr, 5),
                    'p95': np.percentile(arr, 95),
                }

        return uncertainties

    def compute_excavation_depth(self, D: float, crater_type: str = 'simple') -> Dict:
        """
        Compute excavation depth and related parameters.

        The excavation depth is the depth from which most ejecta originates,
        typically shallower than the crater depth.

        Parameters:
        -----------
        D : float
            Crater diameter (m)
        crater_type : str
            'simple' or 'complex'

        Returns:
        --------
        excavation : dict
            Excavation depth, volume, and related parameters
        """
        if crater_type == 'simple':
            # For simple craters: d_exc ≈ 0.1 * D (Melosh 1989)
            d_exc = 0.1 * D
            # Transient depth d_trans ≈ 0.28 * D
            d_trans = 0.28 * D
        else:
            # For complex craters
            d_exc = 0.08 * D
            d_trans = 0.25 * D

        # Excavation volume (paraboloid)
        V_exc = (np.pi / 8) * D**2 * d_exc

        # Maximum excavation depth (at crater center)
        d_exc_max = 1.5 * d_exc

        return {
            'excavation_depth': d_exc,
            'excavation_depth_max': d_exc_max,
            'transient_depth': d_trans,
            'excavation_volume': V_exc,
            'crater_type': crater_type
        }

    def compute_ejecta_volume(self, D: float, d: float,
                             crater_type: str = 'simple') -> Dict:
        """
        Compute ejecta volume and distribution parameters.

        Parameters:
        -----------
        D : float
            Crater diameter (m)
        d : float
            Crater depth (m)
        crater_type : str
            'simple' or 'complex'

        Returns:
        --------
        ejecta : dict
            Ejecta volume, blanket thickness, and range estimates
        """
        # Crater volume
        V_crater = (np.pi / 8) * D**2 * d

        # Excavation calculations
        exc = self.compute_excavation_depth(D, crater_type)
        V_excavated = exc['excavation_volume']

        # Ejecta volume (accounting for bulking factor ~1.2)
        bulking_factor = 1.2
        V_ejecta = V_excavated * bulking_factor

        # Continuous ejecta blanket extends to ~1 crater radius
        ejecta_range_inner = D / 2  # Crater rim
        ejecta_range_outer = 1.5 * D  # Outer edge of continuous ejecta

        # Area of ejecta blanket (annular region)
        A_ejecta = np.pi * (ejecta_range_outer**2 - ejecta_range_inner**2)

        # Average ejecta thickness
        thickness_avg = V_ejecta / A_ejecta

        # Ejecta thickness decreases with distance: t(r) ~ (r/D)^-3
        # At rim (r = D/2): thickness is maximum
        thickness_rim = 0.14 * d  # Empirical relation

        return {
            'ejecta_volume': V_ejecta,
            'excavated_volume': V_excavated,
            'bulking_factor': bulking_factor,
            'ejecta_range_continuous': ejecta_range_outer,
            'ejecta_thickness_avg': thickness_avg,
            'ejecta_thickness_rim': thickness_rim,
            'ejecta_blanket_area': A_ejecta,
        }

    def generate_report(self,
                       diameter: float,
                       depth: float,
                       velocity: float,
                       target_material: str,
                       impactor_material: str,
                       latitude: float,
                       longitude: float,
                       diameter_uncertainty: float = None,
                       depth_uncertainty: float = None,
                       velocity_uncertainty: float = None,
                       crater_type: str = 'simple',
                       output_filename: str = 'impact_analysis_report.pdf',
                       crater_name: str = None) -> str:
        """
        Generate comprehensive PDF report.

        Parameters:
        -----------
        diameter : float
            Crater diameter (m)
        depth : float
            Crater depth (m)
        velocity : float
            Impact velocity (m/s)
        target_material : str
            Target material name (from MATERIALS dict)
        impactor_material : str
            Impactor type (from IMPACTORS dict)
        latitude : float
            Crater latitude (decimal degrees)
        longitude : float
            Crater longitude (decimal degrees)
        diameter_uncertainty : float, optional
            Uncertainty in diameter (m)
        depth_uncertainty : float, optional
            Uncertainty in depth (m)
        velocity_uncertainty : float, optional
            Uncertainty in velocity (m/s)
        crater_type : str
            'simple' or 'complex'
        output_filename : str
            Output PDF filename
        crater_name : str, optional
            Name of the crater

        Returns:
        --------
        output_filename : str
            Path to generated PDF
        """
        # Set default uncertainties if not provided
        if diameter_uncertainty is None:
            diameter_uncertainty = 0.05 * diameter  # 5% default
        if depth_uncertainty is None:
            depth_uncertainty = 0.10 * depth  # 10% default
        if velocity_uncertainty is None:
            velocity_uncertainty = 2000  # ±2 km/s default

        # Create PDF document
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=letter,
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        # Container for elements
        story = []

        # Get materials
        target = MATERIALS[target_material]
        impactor = IMPACTORS[impactor_material]

        # Perform calculations
        calc = ImpactScaling(target, impactor.density)
        results = calc.compute_impactor_params(
            diameter, depth, velocity, crater_type=crater_type
        )

        # Compute uncertainties
        uncertainties = self.compute_uncertainties(
            diameter, depth, velocity,
            diameter_uncertainty, depth_uncertainty, velocity_uncertainty,
            target_material, impactor.density, crater_type
        )

        # Compute excavation and ejecta
        excavation = self.compute_excavation_depth(diameter, crater_type)
        ejecta = self.compute_ejecta_volume(diameter, depth, crater_type)

        # ===== PAGE 1: SUMMARY =====
        story.extend(self._generate_summary_page(
            diameter, depth, velocity, latitude, longitude,
            target, impactor, results, uncertainties,
            excavation, ejecta, crater_type, crater_name,
            diameter_uncertainty, depth_uncertainty, velocity_uncertainty
        ))

        story.append(PageBreak())

        # ===== PAGE 2+: THEORETICAL EXPLANATION =====
        story.extend(self._generate_theory_pages(
            diameter, depth, velocity,
            target, impactor, results,
            excavation, ejecta, crater_type
        ))

        # Build PDF
        doc.build(story)

        return output_filename

    def _generate_summary_page(self, D, d, U, lat, lon,
                               target, impactor, results, uncertainties,
                               excavation, ejecta, crater_type, crater_name,
                               D_err, d_err, U_err):
        """Generate the summary page (page 1)."""
        elements = []

        # Title
        title_text = "IMPACT CRATER ANALYSIS REPORT"
        if crater_name:
            title_text += f"<br/>{crater_name}"
        elements.append(Paragraph(title_text, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.2*inch))

        # Report metadata
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_text = f"<i>Generated: {date_str}</i>"
        elements.append(Paragraph(meta_text, self.styles['Caption']))
        elements.append(Spacer(1, 0.3*inch))

        # === CRATER LOCATION ===
        elements.append(Paragraph("Crater Location", self.styles['SectionHeading']))

        loc_data = [
            ['Latitude:', f'{lat:.6f}°'],
            ['Longitude:', f'{lon:.6f}°'],
            ['Coordinate System:', 'Decimal Degrees']
        ]
        loc_table = Table(loc_data, colWidths=[2*inch, 3*inch])
        loc_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(loc_table)
        elements.append(Spacer(1, 0.2*inch))

        # === OBSERVED CRATER PROPERTIES ===
        elements.append(Paragraph("Observed Crater Properties", self.styles['SectionHeading']))

        obs_data = [
            ['<b>Parameter</b>', '<b>Value</b>', '<b>Uncertainty</b>'],
            ['Crater Diameter', f'{D:.1f} m ({D/1000:.3f} km)', f'± {D_err:.1f} m'],
            ['Crater Depth', f'{d:.1f} m', f'± {d_err:.1f} m'],
            ['Depth/Diameter Ratio', f'{d/D:.3f}', '—'],
            ['Crater Type', crater_type.capitalize(), '—'],
            ['Impact Velocity (assumed)', f'{U/1000:.1f} km/s', f'± {U_err/1000:.1f} km/s'],
        ]
        obs_table = Table(obs_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        obs_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(obs_table)
        elements.append(Spacer(1, 0.2*inch))

        # === COMPUTED IMPACTOR PROPERTIES ===
        elements.append(Paragraph("Computed Impactor Properties", self.styles['SectionHeading']))

        # Format with uncertainties
        L = results['impactor_diameter']
        m = results['impactor_mass']
        E = results['impact_energy']

        if 'impactor_diameter' in uncertainties:
            L_unc = uncertainties['impactor_diameter']
            L_str = f"{L:.2f} m ({L_unc['p16']:.2f} – {L_unc['p84']:.2f})"
        else:
            L_str = f"{L:.2f} m"

        if 'impactor_mass' in uncertainties:
            m_unc = uncertainties['impactor_mass']
            m_str = f"{m:.2e} kg ({m_unc['p16']:.2e} – {m_unc['p84']:.2e})"
        else:
            m_str = f"{m:.2e} kg"

        if 'impact_energy' in uncertainties:
            E_unc = uncertainties['impact_energy']
            E_str = f"{E:.2e} J ({E_unc['p16']:.2e} – {E_unc['p84']:.2e})"
            E_Mt = E / 4.184e15
            E_Mt_str = f"{E_Mt:.2f} Mt TNT"
        else:
            E_str = f"{E:.2e} J"
            E_Mt_str = f"{E/4.184e15:.2f} Mt TNT"

        imp_data = [
            ['<b>Parameter</b>', '<b>Value</b>', '<b>Range (16th–84th percentile)</b>'],
            ['Impactor Diameter', f'{L:.2f} m', L_str.split('(')[1].rstrip(')') if '(' in L_str else '—'],
            ['Impactor Mass', f'{m:.2e} kg', ''],
            ['', f'({m/1e9:.2f} million kg)', ''],
            ['Impact Energy', f'{E:.2e} J', ''],
            ['', E_Mt_str, ''],
            ['Impact Momentum', f'{results["impact_momentum"]:.2e} kg⋅m/s', '—'],
        ]
        imp_table = Table(imp_data, colWidths=[2.2*inch, 2.2*inch, 1.8*inch])
        imp_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(imp_table)
        elements.append(Spacer(1, 0.2*inch))

        # === MATERIAL PROPERTIES ===
        elements.append(Paragraph("Material Properties", self.styles['SectionHeading']))

        mat_data = [
            ['<b>Property</b>', '<b>Target</b>', '<b>Impactor</b>'],
            ['Material', target.name, impactor.name],
            ['Density', f'{target.density:.0f} kg/m³', f'{impactor.density:.0f} kg/m³'],
            ['Strength', f'{target.strength:.2e} Pa', '—'],
            ['Surface Gravity', f'{target.gravity:.2f} m/s²', '—'],
        ]
        mat_table = Table(mat_data, colWidths=[2*inch, 2*inch, 2*inch])
        mat_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(mat_table)
        elements.append(Spacer(1, 0.2*inch))

        # === SCALING ANALYSIS ===
        elements.append(Paragraph("Scaling Regime Analysis", self.styles['SectionHeading']))

        regime_text = f"""
        The impact is in the <b>{results['regime'].upper()} REGIME</b>, meaning that
        {'material strength' if results['regime'] == 'strength' else 'gravity'} is the
        dominant factor controlling crater size. The dimensionless scaling parameters are:
        """
        elements.append(Paragraph(regime_text, self.styles['BodyText']))

        scale_data = [
            ['π₂ (gravity parameter)', f"{results['pi_2']:.2e}", '= gL/U²'],
            ['π₃ (strength parameter)', f"{results['pi_3']:.2e}", '= Y/(ρU²)'],
            ['π₄ (density ratio)', f"{results['pi_4']:.3f}", '= ρ_target/ρ_impactor'],
        ]
        scale_table = Table(scale_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        scale_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'LEFT'),
            ('FONTNAME', (2, 0), (2, -1), 'Courier'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(scale_table)
        elements.append(Spacer(1, 0.2*inch))

        # === EXCAVATION AND EJECTA ===
        elements.append(Paragraph("Excavation and Ejecta Estimates", self.styles['SectionHeading']))

        exc_data = [
            ['<b>Parameter</b>', '<b>Value</b>'],
            ['Excavation Depth', f"{excavation['excavation_depth']:.2f} m"],
            ['Maximum Excavation Depth', f"{excavation['excavation_depth_max']:.2f} m"],
            ['Excavated Volume', f"{excavation['excavation_volume']:.2e} m³"],
            ['Ejecta Volume (with bulking)', f"{ejecta['ejecta_volume']:.2e} m³"],
            ['Continuous Ejecta Range', f"{ejecta['ejecta_range_continuous']:.1f} m"],
            ['Average Ejecta Thickness', f"{ejecta['ejecta_thickness_avg']:.2f} m"],
            ['Ejecta Thickness at Rim', f"{ejecta['ejecta_thickness_rim']:.2f} m"],
        ]
        exc_table = Table(exc_data, colWidths=[3.5*inch, 2.5*inch])
        exc_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(exc_table)

        return elements

    def _generate_theory_pages(self, D, d, U, target, impactor,
                               results, excavation, ejecta, crater_type):
        """Generate theoretical explanation pages (page 2+)."""
        elements = []

        # Title for theory section
        elements.append(Paragraph("THEORETICAL METHODOLOGY", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.2*inch))

        # === SECTION 1: OVERVIEW ===
        elements.append(Paragraph("1. Overview of Inverse Scaling Method", self.styles['SectionHeading']))

        overview_text = """
        This analysis uses <b>pi-group scaling relationships</b> developed by Holsapple (1993)
        and refined by subsequent researchers. The method performs <i>inverse calculations</i>:
        given an observed crater size, we compute the impactor properties that created it.
        <br/><br/>
        The fundamental challenge is that crater size depends on multiple parameters: impactor
        size, velocity, density, target properties, and impact angle. By assuming vertical
        impact and a known velocity, we can uniquely determine the impactor size.
        """
        elements.append(Paragraph(overview_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 2: PI-GROUP SCALING ===
        elements.append(Paragraph("2. Pi-Group Scaling Theory", self.styles['SectionHeading']))

        pigroup_text = """
        Impact cratering can be described by three dimensionless parameters (π-groups):
        """
        elements.append(Paragraph(pigroup_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))

        # Equations for pi-groups
        elements.append(Paragraph("<b>Gravity Parameter:</b>", self.styles['SubsectionHeading']))
        eq1 = f"π₂ = gL/U²"
        elements.append(Paragraph(eq1, self.styles['Equation']))

        pi2_text = f"""
        Where g = {target.gravity} m/s² (surface gravity),
        L = {2*results['impactor_radius']:.2f} m (impactor diameter),
        U = {U} m/s (impact velocity).
        <br/>
        Computed value: <b>π₂ = {results['pi_2']:.2e}</b>
        """
        elements.append(Paragraph(pi2_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))

        elements.append(Paragraph("<b>Strength Parameter:</b>", self.styles['SubsectionHeading']))
        eq2 = f"π₃ = Y/(ρU²)"
        elements.append(Paragraph(eq2, self.styles['Equation']))

        pi3_text = f"""
        Where Y = {target.strength:.2e} Pa (target strength),
        ρ = {target.density} kg/m³ (target density),
        U = {U} m/s (impact velocity).
        <br/>
        Computed value: <b>π₃ = {results['pi_3']:.2e}</b>
        """
        elements.append(Paragraph(pi3_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))

        elements.append(Paragraph("<b>Density Ratio:</b>", self.styles['SubsectionHeading']))
        eq3 = f"π₄ = ρ_target / ρ_impactor"
        elements.append(Paragraph(eq3, self.styles['Equation']))

        pi4_text = f"""
        Where ρ_target = {target.density} kg/m³,
        ρ_impactor = {impactor.density} kg/m³.
        <br/>
        Computed value: <b>π₄ = {results['pi_4']:.3f}</b>
        """
        elements.append(Paragraph(pi4_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 3: REGIME DETERMINATION ===
        elements.append(Paragraph("3. Scaling Regime Determination", self.styles['SectionHeading']))

        regime_determination = f"""
        The dominant scaling regime is determined by comparing π₂ and π₃:
        <br/><br/>
        • If <b>π₃ &gt; π₂</b>: Strength regime (material strength controls crater size)
        <br/>
        • If <b>π₂ &gt; π₃</b>: Gravity regime (gravity controls crater size)
        <br/><br/>
        For this impact:
        <br/>
        π₃ = {results['pi_3']:.2e}, π₂ = {results['pi_2']:.2e}
        <br/><br/>
        Since π₃ {'>' if results['regime'] == 'strength' else '<'} π₂,
        the impact is in the <b>{results['regime'].upper()} REGIME</b>.
        """
        elements.append(Paragraph(regime_determination, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 4: SCALING LAW APPLICATION ===
        elements.append(Paragraph("4. Crater Diameter Scaling Law", self.styles['SectionHeading']))

        L_imp = results['impactor_diameter']
        mu = target.mu
        nu = target.nu

        if results['regime'] == 'strength':
            scaling_eq = f"D/L = K₁ × (ρ_imp/ρ_target)^(1/3) × (ρU²/Y)^(μ/(2+μ))"
            elements.append(Paragraph("<b>Strength Regime Scaling:</b>", self.styles['SubsectionHeading']))
            elements.append(Paragraph(scaling_eq, self.styles['Equation']))

            exponent = mu / (2.0 + mu)
            scaling_explanation = f"""
            Where:
            <br/>• K₁ = 1.25 (empirical coupling parameter)
            <br/>• μ = {mu} (strength scaling exponent for {target.name})
            <br/>• Exponent μ/(2+μ) = {exponent:.3f}
            <br/><br/>
            Substituting values:
            <br/>• ρ_imp/ρ_target = {impactor.density}/{target.density} = {impactor.density/target.density:.3f}
            <br/>• ρU²/Y = ({target.density} × {U}²) / {target.strength:.2e} = {(target.density * U**2)/target.strength:.2e}
            <br/>• (ρU²/Y)^{exponent:.3f} = {((target.density * U**2)/target.strength)**exponent:.2e}
            """
        else:
            scaling_eq = f"D/L = K₂ × (ρ_imp/ρ_target)^(1/3) × (U²/gL)^(ν/(2+ν))"
            elements.append(Paragraph("<b>Gravity Regime Scaling:</b>", self.styles['SubsectionHeading']))
            elements.append(Paragraph(scaling_eq, self.styles['Equation']))

            exponent = nu / (2.0 + nu)
            scaling_explanation = f"""
            Where:
            <br/>• K₂ = 1.61 (empirical coupling parameter)
            <br/>• ν = {nu} (gravity scaling exponent for {target.name})
            <br/>• Exponent ν/(2+ν) = {exponent:.3f}
            <br/><br/>
            Substituting values:
            <br/>• ρ_imp/ρ_target = {impactor.density}/{target.density} = {impactor.density/target.density:.3f}
            <br/>• U²/gL = {U}² / ({target.gravity} × {L_imp:.2f}) = {U**2/(target.gravity*L_imp):.2e}
            <br/>• (U²/gL)^{exponent:.3f} = {(U**2/(target.gravity*L_imp))**exponent:.2e}
            """

        elements.append(Paragraph(scaling_explanation, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 5: TRANSIENT TO FINAL CRATER ===
        elements.append(Paragraph("5. Transient to Final Crater Conversion", self.styles['SectionHeading']))

        D_trans = results['crater_diameter_transient']
        D_final = results['crater_diameter_final']

        transient_text = f"""
        The scaling law predicts the <b>transient crater</b> size (the initial cavity before
        modification). For {crater_type} craters, the final crater undergoes:
        <br/><br/>
        """

        if crater_type == 'simple':
            transient_text += f"""
            • Modest rim collapse and slumping
            <br/>• Diameter enlargement: D_final ≈ 1.25 × D_transient
            <br/>• Depth decrease due to breccia lens formation
            <br/><br/>
            For this crater:
            <br/>• D_transient = {D_trans:.2f} m
            <br/>• D_final = {D_final:.2f} m (observed)
            <br/>• Expansion factor = {D_final/D_trans:.2f}
            """
        else:
            transient_text += f"""
            • Significant collapse with central peak formation
            <br/>• Terraced wall formation
            <br/>• Diameter enlargement: D_final ≈ 1.3 × D_transient
            <br/>• Substantial depth reduction (d/D drops from ~0.28 to ~0.1)
            <br/><br/>
            For this crater:
            <br/>• D_transient = {D_trans:.2f} m
            <br/>• D_final = {D_final:.2f} m (observed)
            <br/>• Expansion factor = {D_final/D_trans:.2f}
            """

        elements.append(Paragraph(transient_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 6: EXCAVATION DEPTH ===
        elements.append(PageBreak())
        elements.append(Paragraph("6. Excavation Depth Calculations", self.styles['SectionHeading']))

        exc_text = f"""
        The <b>excavation depth</b> represents the depth from which most ejecta originates.
        It is shallower than the transient crater depth due to the geometry of the
        expanding flow field.
        <br/><br/>
        For {crater_type} craters, empirical relations give:
        <br/>• d_excavation ≈ {0.1 if crater_type=='simple' else 0.08} × D_final
        <br/><br/>
        Calculations:
        <br/>• D_final = {D_final:.2f} m
        <br/>• d_excavation = {excavation['excavation_depth']:.2f} m
        <br/>• d_excavation_max (at center) = {excavation['excavation_depth_max']:.2f} m
        <br/>• V_excavation = (π/8) × D² × d_exc = {excavation['excavation_volume']:.2e} m³
        <br/><br/>
        The excavation depth varies with distance from crater center as:
        <br/><br/>
        d_exc(r) = d_exc_max × [1 - (r/R)²]
        <br/><br/>
        where R = D/2 is the crater radius and r is the radial distance from center.
        """
        elements.append(Paragraph(exc_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 7: EJECTA VOLUME ===
        elements.append(Paragraph("7. Ejecta Volume and Distribution", self.styles['SectionHeading']))

        ejecta_text = f"""
        The ejecta volume is computed from the excavation volume with a correction for
        <b>bulking</b> (volume expansion due to fracturing and void space):
        <br/><br/>
        V_ejecta = f_bulk × V_excavation
        <br/><br/>
        Where f_bulk ≈ {ejecta['bulking_factor']} for fractured rock/regolith.
        <br/><br/>
        <b>Computed Values:</b>
        <br/>• V_excavation = {ejecta['excavated_volume']:.2e} m³
        <br/>• V_ejecta = {ejecta['ejecta_volume']:.2e} m³
        <br/><br/>
        <b>Ejecta Blanket Distribution:</b>
        <br/>
        The continuous ejecta blanket extends from the crater rim to approximately
        1.5× the crater diameter:
        <br/>• Inner radius (rim): {D/2:.1f} m
        <br/>• Outer radius: {ejecta['ejecta_range_continuous']:.1f} m
        <br/>• Blanket area: {ejecta['ejecta_blanket_area']:.2e} m²
        <br/>• Average thickness: {ejecta['ejecta_thickness_avg']:.2f} m
        <br/>• Thickness at rim: {ejecta['ejecta_thickness_rim']:.2f} m
        <br/><br/>
        Ejecta thickness decreases with distance following a power law:
        <br/><br/>
        t(r) ≈ t_rim × (r/r_rim)^(-3)
        <br/><br/>
        where r is the distance from the crater center.
        """
        elements.append(Paragraph(ejecta_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 8: ASSUMPTIONS ===
        elements.append(Paragraph("8. Key Assumptions and Limitations", self.styles['SectionHeading']))

        assumptions_text = f"""
        This analysis relies on the following assumptions:
        <br/><br/>
        <b>1. Vertical Impact:</b> Impact angle assumed to be 90° (vertical). Oblique impacts
        (θ &lt; 45°) produce elongated craters and require additional corrections.
        <br/><br/>
        <b>2. Impact Velocity:</b> Assumed velocity of {U/1000:.1f} km/s with uncertainty
        ±{2000/1000:.1f} km/s. Actual velocity can vary significantly based on impactor origin
        (asteroid vs. comet).
        <br/><br/>
        <b>3. Homogeneous Target:</b> Target assumed to be homogeneous {target.name}.
        Real targets often have layering, which affects crater formation.
        <br/><br/>
        <b>4. Fresh Crater:</b> Analysis assumes minimal degradation. Degraded craters require
        corrections for infilling and erosion.
        <br/><br/>
        <b>5. Strength and Density:</b> Target strength ({target.strength:.2e} Pa) and density
        ({target.density} kg/m³) are nominal values that can vary spatially.
        <br/><br/>
        <b>6. Scaling Law Validity:</b> Scaling laws are calibrated from laboratory experiments
        and numerical simulations. Extrapolation to very large or small scales introduces uncertainty.
        <br/><br/>
        <b>7. Single Impactor:</b> Assumes single impact event. Multiple or sequential impacts
        complicate the analysis.
        """
        elements.append(Paragraph(assumptions_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 9: UNCERTAINTY SOURCES ===
        elements.append(Paragraph("9. Sources of Uncertainty", self.styles['SectionHeading']))

        uncertainty_text = """
        The uncertainties in computed impactor properties arise from:
        <br/><br/>
        <b>Measurement Uncertainties:</b>
        <br/>• Crater diameter and depth measurements (topographic data resolution)
        <br/>• Identification of crater rim position
        <br/>• Depth measurement affected by infill and degradation
        <br/><br/>
        <b>Physical Uncertainties:</b>
        <br/>• Impact velocity (largest source of uncertainty, typically ±20-30%)
        <br/>• Impact angle (if non-vertical)
        <br/>• Target material properties (strength, porosity, layering)
        <br/>• Impactor density (varies by composition)
        <br/><br/>
        <b>Model Uncertainties:</b>
        <br/>• Scaling law parameters (empirical coefficients)
        <br/>• Transient-to-final crater conversion
        <br/>• Regime transition (strength vs. gravity)
        <br/><br/>
        Monte Carlo analysis was performed to propagate these uncertainties through the
        calculations, producing the uncertainty ranges shown in the summary table.
        """
        elements.append(Paragraph(uncertainty_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # === SECTION 10: REFERENCES ===
        elements.append(Paragraph("10. References", self.styles['SectionHeading']))

        references_text = """
        <b>Holsapple, K.A. (1993).</b> "The scaling of impact processes in planetary sciences."
        <i>Annual Review of Earth and Planetary Sciences</i>, 21, 333-373.
        <br/><br/>
        <b>Holsapple, K.A., & Housen, K.R. (2007).</b> "A crater and its ejecta: An interpretation
        of Deep Impact." <i>Icarus</i>, 187, 345-356.
        <br/><br/>
        <b>Collins, G.S., Melosh, H.J., & Marcus, R.A. (2005).</b> "Earth Impact Effects Program."
        <i>Meteoritics & Planetary Science</i>, 40, 817-840.
        <br/><br/>
        <b>Melosh, H.J. (1989).</b> <i>Impact Cratering: A Geologic Process.</i>
        Oxford University Press.
        <br/><br/>
        <b>Schmidt, R.M., & Housen, K.R. (1987).</b> "Some recent advances in the scaling of
        impact and explosion cratering." <i>International Journal of Impact Engineering</i>,
        5, 543-560.
        """
        elements.append(Paragraph(references_text, self.styles['BodyText']))

        return elements


def generate_impact_report(diameter, depth, velocity, target_material,
                          impactor_material, latitude, longitude,
                          **kwargs):
    """
    Convenience function to generate impact report.

    Parameters:
    -----------
    diameter : float
        Crater diameter (m)
    depth : float
        Crater depth (m)
    velocity : float
        Impact velocity (m/s)
    target_material : str
        Target material name from MATERIALS dict
    impactor_material : str
        Impactor type from IMPACTORS dict
    latitude : float
        Crater latitude (degrees)
    longitude : float
        Crater longitude (degrees)
    **kwargs : optional
        Additional parameters (diameter_uncertainty, depth_uncertainty, etc.)

    Returns:
    --------
    filename : str
        Path to generated PDF report
    """
    generator = ImpactReportGenerator()
    return generator.generate_report(
        diameter, depth, velocity,
        target_material, impactor_material,
        latitude, longitude,
        **kwargs
    )


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate impact crater analysis report'
    )
    parser.add_argument('--diameter', type=float, required=True,
                       help='Crater diameter (m)')
    parser.add_argument('--depth', type=float, required=True,
                       help='Crater depth (m)')
    parser.add_argument('--velocity', type=float, required=True,
                       help='Impact velocity (m/s)')
    parser.add_argument('--target', type=str, required=True,
                       help='Target material (e.g., lunar_regolith)')
    parser.add_argument('--impactor', type=str, required=True,
                       help='Impactor type (e.g., asteroid_rock)')
    parser.add_argument('--latitude', type=float, required=True,
                       help='Crater latitude (degrees)')
    parser.add_argument('--longitude', type=float, required=True,
                       help='Crater longitude (degrees)')
    parser.add_argument('--output', type=str, default='impact_report.pdf',
                       help='Output PDF filename')
    parser.add_argument('--crater-name', type=str, default=None,
                       help='Optional crater name')
    parser.add_argument('--crater-type', type=str, default='simple',
                       choices=['simple', 'complex'],
                       help='Crater type (simple or complex)')

    args = parser.parse_args()

    print("Generating impact crater analysis report...")
    print(f"  Diameter: {args.diameter} m")
    print(f"  Depth: {args.depth} m")
    print(f"  Velocity: {args.velocity} m/s")
    print(f"  Location: {args.latitude}°, {args.longitude}°")

    filename = generate_impact_report(
        diameter=args.diameter,
        depth=args.depth,
        velocity=args.velocity,
        target_material=args.target,
        impactor_material=args.impactor,
        latitude=args.latitude,
        longitude=args.longitude,
        output_filename=args.output,
        crater_name=args.crater_name,
        crater_type=args.crater_type
    )

    print(f"\nReport generated: {filename}")
