import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyparsing import alphas


class FDTD2D_TMz:
    """
    2D FDTD для TMz хвиль (Ez, Hx, Hy)
    Без провідності, зі струмовим джерелом Jz
    """

    def __init__(self, Nx=120, Ny=120, dx=1e-3, dy=1e-3, Nt=600):
        # --- Параметри сітки ---
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = dx, dy
        S = 1
        self.dt = S / (np.sqrt((1/dx**2) + (1/dy**2)))
        self.Nt = Nt

        # Courant numbers
        self.Sx = self.dt / self.dx
        self.Sy = self.dt / self.dy

        # --- Поля (Yee-схема) ---
        self.Ez = np.zeros((Nx, Ny))
        self.Hx = np.zeros((Nx, Ny-1))
        self.Hy = np.zeros((Nx-1, Ny))

        self.Ez_prev = np.zeros_like(self.Ez)

        # --- Джерело ---
        self.src_x, self.src_y = Nx//2, Ny//2
        self.t0, self.spread = 100 * self.dt, 30 * self.dt

        # --- Датчик ---
        self.probe_x, self.probe_y = Nx//2 + 20, Ny//2
        self.signal = []

        # gap
        self.gap_position = 80
        self.gap_width = 10

        # TFSF
        self.left_i = 20
        self.right_i = 100
        self.left_j = 20
        self.right_j = 100
        self.tfsf_i = np.arange(self.left_i, self.right_i + 1)
        self.tfsf_j = np.arange(self.left_j, self.right_j + 1)

        self.theta = 0

        self.eps = np.ones((Nx, Ny))

        self.alpha = 0.95 * np.pi / 2
        self.margin = 5

    # -------------------------------------------------
    def calc_incident_Ez(self, n, X, Y):
        """Падаюча E-компонента"""
        s = (X * np.cos(self.theta) + Y * np.sin(self.theta))
        t_eff = n * self.dt - s
        return np.exp(-((t_eff - self.t0) / self.spread) ** 2)

    def calc_incident_Hx(self, n, X, Y):
        """Падаюча Hx-компонента"""
        s = (X * np.cos(self.theta) + Y * np.sin(self.theta))
        t_eff = n * self.dt - s
        return np.sin(self.theta) * np.exp(-((t_eff - self.t0) / self.spread) ** 2)

    def calc_incident_Hy(self, n, X, Y):
        """Падаюча Hy-компонента"""
        s = (X * np.cos(self.theta) + Y * np.sin(self.theta))
        t_eff = n * self.dt - s
        return -np.cos(self.theta) * np.exp(-((t_eff - self.t0) / self.spread) ** 2)

    # -------------------------------------------------
    def Jz(self, n):
        """Гаусовий імпульс струму"""
        return np.exp(-((n - self.t0) / self.spread)**2)

    # -------------------------------------------------
    def update_H(self):
        """Оновлення Hx, Hy"""
        self.Hx -= self.Sy * (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy += self.Sx * (self.Ez[1:, :] - self.Ez[:-1, :])

    # -------------------------------------------------
    def update_E(self):
        """Оновлення Ez (внутрішні вузли)"""
        curl_H = (
            (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx
            - (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
        )
        self.Ez[1:-1, 1:-1] += (self.dt / self.eps[1:-1, 1:-1]) * curl_H

    # -------------------------------------------------
    def correction_H(self, n):
        """TFSF корекція для H-полів"""
        # Ліва межа
        self.Hy[self.left_i - 1, self.tfsf_j] -= (self.dt / self.dx) * \
                                                     self.calc_incident_Ez(n, self.left_i * self.dx,
                                                                           self.tfsf_j * self.dy)

        # Права межа
        self.Hy[self.right_i, self.tfsf_j] += (self.dt / self.dx) * \
                                              self.calc_incident_Ez(n, self.right_i * self.dx,
                                                                    self.tfsf_j * self.dy)

        # Нижня межа
        self.Hx[self.tfsf_i, self.left_j - 1] += (self.dt / self.dy) * \
                                                     self.calc_incident_Ez(n, self.tfsf_i * self.dx,
                                                                           self.left_j * self.dy)

        # Верхня межа
        self.Hx[self.tfsf_i, self.right_j] -= (self.dt / self.dy) * \
                                                  self.calc_incident_Ez(n, self.tfsf_i * self.dx,
                                                                        self.right_j * self.dy)

    # -------------------------------------------------
    def correction_E(self, n):
        """TFSF корекція для E-полів"""
        # Ліва межа
        self.Ez[self.left_i, self.tfsf_j] -= (self.dt / self.dx) * \
                                             self.calc_incident_Hy(n + 0.5, (self.left_i - 0.5) * self.dx,
                                                                   self.tfsf_j * self.dy)

        # Права межа
        self.Ez[self.right_i, self.tfsf_j] += (self.dt / self.dx) * \
                                              self.calc_incident_Hy(n + 0.5, (self.right_i + 0.5) * self.dx,
                                                                    self.tfsf_j * self.dy)

        # Нижня межа
        self.Ez[self.tfsf_i, self.left_j] += (self.dt / self.dy) * \
                                             self.calc_incident_Hx(n + 0.5, self.tfsf_i * self.dx,
                                                                   (self.left_j - 0.5) * self.dy)

        # Верхня межа
        self.Ez[self.tfsf_i, self.right_j] -= (self.dt / self.dy) * \
                                              self.calc_incident_Hx(n + 0.5, self.tfsf_i * self.dx,
                                                                    (self.right_j + 0.5) * self.dy)

    # -------------------------------------------------
    def apply_mur_abc(self):
        """Mur ABC (1-й порядок)"""
        cx = (self.Sx - 1) / (self.Sx + 1)
        cy = (self.Sy - 1) / (self.Sy + 1)

        # left
        self.Ez[0, 1:-1] = (
            self.Ez_prev[1, 1:-1]
            + cx * (self.Ez[1, 1:-1] - self.Ez_prev[0, 1:-1])
        )
        # right
        self.Ez[-1, 1:-1] = (
            self.Ez_prev[-2, 1:-1]
            + cx * (self.Ez[-2, 1:-1] - self.Ez_prev[-1, 1:-1])
        )

        # bottom
        self.Ez[1:-1, 0] = (
                self.Ez_prev[1:-1, 1]
                + cy * (self.Ez[1:-1, 1] - self.Ez_prev[1:-1, 0])
        )
        # top
        self.Ez[1:-1, -1] = (
                self.Ez_prev[1:-1, -2]
                + cy * (self.Ez[1:-1, -2] - self.Ez_prev[1:-1, -1])
        )

        # кути
        self.Ez[0, 0] = self.Ez[1, 1]
        self.Ez[0, -1] = self.Ez[1, -2]
        self.Ez[-1, 0] = self.Ez[-2, 1]
        self.Ez[-1, -1] = self.Ez[-2, -2]


    # -------------------------------------------------
    def scatterer(self):
        self.eps[55:65, 55:65] = 1
        j1, j2 = self.left_j - self.margin, self.right_j + self.margin
        j_av = (j1 + j2) // 2
        for j in range(j1, j_av + 1):
            x = self.right_i - self.margin - (j_av - j) * np.tan(np.pi/2 - self.alpha/2)
            self.Ez[round(x), j] = 0
        for j in range(j_av + 1, j2 + 1):
            x = self.right_i - self.margin - (j - j_av) * np.tan(np.pi/2 - self.alpha/2)
            self.Ez[round(x), j] = 0
    # -------------------------------------------------
    def source(self, n):
        """Струмове джерело Jz"""
        self.Ez[self.src_x, self.src_y] -= self.dt * self.Jz(n)

    # -------------------------------------------------
    def step(self, n):
        """Один часовий крок"""
        self.Ez_prev[:, :] = self.Ez
        self.update_H()
        self.correction_H(n)
        self.update_E()
        self.correction_E(n)
        self.apply_mur_abc()
        self.scatterer()
        self.signal.append(self.Ez[self.probe_x, self.probe_y])

    # -------------------------------------------------
    def animate(self):
        """Анімація поширення поля"""

        fig, ax = plt.subplots()
        v_amp = 1e4
        im = ax.imshow(self.Ez.T, cmap="RdBu_r",
                       vmin=-v_amp, vmax=v_amp, origin="lower")
        y = (self.left_j + self.right_j) / 2
        x = self.margin + (y - self.left_j + self.margin) * np.tan(np.pi/2 - self.alpha/2)
        ax.plot((self.right_i - x , self.right_i - self.margin), (self.left_j - self.margin, y), c="k", linewidth=2)
        ax.plot((self.right_i - x , self.right_i - self.margin), (self.right_j  + self.margin, y), c="k", linewidth=2)
        ax.plot((self.left_i, self.right_i), (self.left_j, self.left_j), c="g", linestyle="--", linewidth=1)
        ax.plot((self.left_i, self.right_i), (self.right_j, self.right_j), c="g", linestyle="--", linewidth=1)
        ax.plot((self.left_i, self.left_i), (self.left_j, self.right_j), c="g", linestyle="--", linewidth=1)
        ax.plot((self.right_i, self.right_j), (self.left_j, self.right_j), c="g", linestyle="--", linewidth=1)
        plt.colorbar(im, ax=ax)
        ax.set_title("2D TMz FDTD: Ez(x,y)")

        def update(n):
            self.step(n)
            im.set_array(15000 * self.Ez.T)
            ax.set_title(f"2D TMz FDTD: Ez(x,y), n = {n}")
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=self.Nt, interval=150, repeat=False
        )
        plt.show()


# =====================================================
# ЗАПУСК ЛАБОРАТОРНОЇ
# =====================================================

if __name__ == "__main__":
    sim = FDTD2D_TMz(Nx=120, Ny=120, Nt=1200)
    sim.animate()
    sim.plot_incident()
